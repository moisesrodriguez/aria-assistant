import os
import json
import re
import ast
import math
import gradio as gr
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_HISTORY_MESSAGES = 20
MAX_TOKENS = 1024

client = InferenceClient(token=HF_TOKEN)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres Aria, una asistente de IA útil, amigable y precisa. Respondes en el mismo idioma que el usuario.

Tienes acceso a las siguientes herramientas. Úsalas cuando el usuario necesite información actual o cálculos matemáticos:

HERRAMIENTA 1: web_search
- Para buscar información actualizada, eventos recientes, noticias, o cualquier hecho que pueda haber cambiado.
- Sintaxis: <tool_call>{"name": "web_search", "arguments": {"query": "texto de búsqueda"}}</tool_call>

HERRAMIENTA 2: calculator
- Para calcular expresiones matemáticas: aritmética, trigonometría, logaritmos, potencias.
- Funciones disponibles: sqrt(), sin(), cos(), tan(), log(), log10(), abs(), ceil(), floor(), exp()
- Constantes: pi, e
- Sintaxis: <tool_call>{"name": "calculator", "arguments": {"expression": "expresión matemática"}}</tool_call>

INSTRUCCIONES DE USO:
1. Si necesitas una herramienta, responde ÚNICAMENTE con el tag <tool_call> en tu respuesta. No añadas texto antes ni después.
2. Solo usa UNA herramienta por respuesta.
3. Cuando recibas el resultado de la herramienta, responde normalmente sin usar otro tool_call.
4. Si no necesitas ninguna herramienta, responde directamente sin usar ningún tag.

Ejemplos correctos:
- Usuario pregunta por noticias recientes → <tool_call>{"name": "web_search", "arguments": {"query": "noticias recientes"}}</tool_call>
- Usuario pide calcular algo → <tool_call>{"name": "calculator", "arguments": {"expression": "sqrt(144)"}}</tool_call>
- Usuario saluda → Responder directamente sin herramienta
"""

# ---------------------------------------------------------------------------
# Safe calculator (AST-based, no raw eval)
# ---------------------------------------------------------------------------
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.FloorDiv, ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load,
)

SAFE_NAMES = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def safe_eval_math(expression: str) -> str:
    try:
        expression = expression.strip().replace("^", "**")
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return "Error: operación no permitida en la expresión."
        code = compile(tree, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, SAFE_NAMES)  # noqa: S307
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return str(round(result, 10))
        return str(result)
    except ZeroDivisionError:
        return "Error: División por cero."
    except Exception as exc:
        return f"Error: No se pudo evaluar '{expression}' — {exc}"


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No se encontraron resultados para esa búsqueda."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Sin título")
            body = r.get("body", "")[:250]
            href = r.get("href", "")
            lines.append(f"{i}. **{title}**\n   {body}\n   Fuente: {href}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Error en la búsqueda: {exc}. Intenta reformular tu pregunta."


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def execute_tool(name: str, arguments: dict) -> str:
    if name == "web_search":
        query = arguments.get("query", "").strip()
        return web_search(query) if query else "Error: consulta vacía."
    if name == "calculator":
        expression = arguments.get("expression", "").strip()
        return safe_eval_math(expression) if expression else "Error: expresión vacía."
    return f"Error: herramienta desconocida '{name}'."


# ---------------------------------------------------------------------------
# Tool call parser — handles both our <tool_call> markers and native tool_calls
# ---------------------------------------------------------------------------
_TOOL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> tuple:
    """Returns (tool_call_dict, text_before_marker) or (None, full_text)."""
    match = _TOOL_PATTERN.search(text)
    if not match:
        return None, text
    try:
        tool_call = json.loads(match.group(1).strip())
        if "name" not in tool_call or "arguments" not in tool_call:
            return None, text
        return tool_call, text[: match.start()].strip()
    except json.JSONDecodeError:
        return None, text


def extract_native_tool_call(response) -> dict | None:
    """Extract tool call from native tool_calls field (model ignores our markers)."""
    try:
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            tc = tool_calls[0]
            return {
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Content extractor — handles both plain strings and Gradio content blocks
# ---------------------------------------------------------------------------
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return " ".join(filter(None, parts))
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------
def build_messages(history: list, user_message: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = history[-MAX_HISTORY_MESSAGES:]
    for msg in recent:
        text = extract_text(msg.get("content", ""))
        if text:
            messages.append({"role": msg["role"], "content": text})
    messages.append({"role": "user", "content": user_message})
    return messages


def call_model(messages: list, stream: bool = False, model: str = MODEL_ID):
    return client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def chat(message: str, history: list):
    messages = build_messages(history, message)

    # --- First pass: detect tool call ---
    try:
        response = call_model(messages, stream=False)
        msg_obj = response.choices[0].message
        assistant_text = (msg_obj.content or "").strip()

        # Some models return tool calls in the native field instead of text
        if not assistant_text:
            native = extract_native_tool_call(response)
            if native:
                assistant_text = f'<tool_call>{json.dumps(native)}</tool_call>'
            else:
                yield "⚠️ El modelo no generó respuesta. Por favor intenta de nuevo."
                return

    except Exception as exc:
        err = str(exc)
        if "429" in err or "rate" in err.lower():
            yield "⏳ Demasiadas solicitudes. Espera un momento e intenta de nuevo."
        elif "401" in err or "token" in err.lower():
            yield "🔑 Error de autenticación. Verifica que HF_TOKEN esté configurado."
        else:
            yield f"❌ Error al conectar con el modelo: {err[:150]}"
        return

    tool_call, text_before = parse_tool_call(assistant_text)

    # --- No tool needed ---
    if not tool_call:
        yield assistant_text
        return

    # --- Tool call detected ---
    tool_name = tool_call["name"]
    tool_indicator = f"🔧 *Consultando {tool_name}...*"
    if text_before:
        tool_indicator = text_before + "\n\n" + tool_indicator
    yield tool_indicator

    tool_result = execute_tool(tool_name, tool_call["arguments"])

    follow_up_messages = messages + [
        {"role": "assistant", "content": assistant_text},
        {
            "role": "user",
            "content": (
                f"[Resultado de {tool_name}]:\n{tool_result}\n\n"
                "Ahora responde al usuario de forma natural usando esta información. "
                "No uses ningún tool_call."
            ),
        },
    ]

    # --- Second pass: stream the final answer ---
    try:
        streamed = ""
        prefix = tool_indicator + "\n\n"
        for chunk in call_model(follow_up_messages, stream=True):
            delta = chunk.choices[0].delta.content or ""
            streamed += delta
            yield prefix + streamed

        if not streamed.strip():
            yield prefix + "Encontré información pero no pude formular una respuesta. Intenta de nuevo."

    except Exception as exc:
        err = str(exc)
        yield tool_indicator + f"\n\n❌ Error al generar la respuesta final: {err[:120]}"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["Hola, ¿quién eres y qué puedes hacer?"],
    ["¿Cuáles son las últimas noticias sobre inteligencia artificial?"],
    ["Calcula la raíz cuadrada de 144 más 37 multiplicado por 2"],
    ["¿Cuánto es el 15% de 2450?"],
    ["Busca información sobre el cambio climático en 2025"],
    ["Calcula sin(pi/4) * cos(pi/3)"],
    ["¿Cuánto es 2 elevado a la potencia 10?"],
    ["¿Quién ganó la última Champions League?"],
]

CSS = """
.gradio-container { max-width: 850px !important; margin: auto !important; }
footer { display: none !important; }
"""

demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🤖 Aria — Agente Conversacional con IA",
    description=(
        "**Aria** es una asistente de IA avanzada construida con modelos de HuggingFace. "
        "Cuenta con **memoria de conversación**, puede **buscar en la web** y realizar "
        "**cálculos matemáticos**.\n\n"
        "**Herramientas disponibles:**\n"
        "- 🔍 **Búsqueda web** — Información actualizada sobre cualquier tema\n"
        "- 🧮 **Calculadora** — Aritmética, trigonometría, logaritmos y más\n\n"
        "*¡Prueba los ejemplos de abajo o escribe lo que necesites!*"
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=CSS,
)

if __name__ == "__main__":
    demo.launch()
