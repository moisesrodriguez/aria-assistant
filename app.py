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
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
FALLBACK_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_HISTORY_MESSAGES = 20
MAX_TOKENS = 1024

client = InferenceClient(token=HF_TOKEN)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Aria, a helpful and knowledgeable AI assistant. You are friendly, concise, and accurate.

You have access to the following tools:

1. **web_search(query)** — Search the web for current information. Use this when users ask about recent events, facts you're unsure about, or anything that benefits from up-to-date information.
2. **calculator(expression)** — Evaluate mathematical expressions. Use this for any arithmetic, algebra, or mathematical computation. Supports: +, -, *, /, **, sqrt(), sin(), cos(), tan(), log(), pi, e.

When you need to use a tool, respond with EXACTLY this format on its own line:
<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

Examples:
<tool_call>{"name": "web_search", "arguments": {"query": "latest AI news 2025"}}</tool_call>
<tool_call>{"name": "calculator", "arguments": {"expression": "sqrt(144) + 37 * 2"}}</tool_call>

RULES:
- Use only ONE tool call per response.
- After receiving tool results, give a natural, helpful answer using the information.
- If you don't need a tool, respond normally without any tool_call tag.
- For the calculator, use Python math syntax (** for power, e.g. sqrt(x), sin(x)).
- Always be concise and clear.
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
                return f"Error: operación no permitida en la expresión."
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
        if not query:
            return "Error: No se proporcionó una consulta de búsqueda."
        return web_search(query)
    if name == "calculator":
        expression = arguments.get("expression", "").strip()
        if not expression:
            return "Error: No se proporcionó una expresión matemática."
        return safe_eval_math(expression)
    return f"Error: Herramienta desconocida '{name}'."


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------
_TOOL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> tuple:
    match = _TOOL_PATTERN.search(text)
    if not match:
        return None, text
    try:
        tool_call = json.loads(match.group(1).strip())
        if "name" not in tool_call or "arguments" not in tool_call:
            return None, text
        text_before = text[: match.start()].strip()
        return tool_call, text_before
    except json.JSONDecodeError:
        return None, text


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def build_messages(history: list, user_message: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})
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


def chat(message: str, history: list):
    messages = build_messages(history, message)

    try:
        # First pass: non-streaming to detect tool calls reliably
        response = call_model(messages, stream=False)
        assistant_text = response.choices[0].message.content or ""

        tool_call, text_before = parse_tool_call(assistant_text)

        if tool_call:
            # Show "thinking" indicator while the tool runs
            thinking = (text_before + "\n\n" if text_before else "") + f"🔧 *Usando {tool_call['name']}...*"
            yield thinking

            tool_result = execute_tool(tool_call["name"], tool_call["arguments"])

            # Feed tool result back for a final streamed response
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({
                "role": "user",
                "content": (
                    f"[Resultado de {tool_call['name']}]:\n{tool_result}\n\n"
                    "Por favor, proporciona una respuesta útil basada en esta información. "
                    "No uses otra llamada a herramienta."
                ),
            })

            streamed = ""
            for chunk in call_model(messages, stream=True):
                delta = chunk.choices[0].delta.content or ""
                streamed += delta
                yield thinking + "\n\n" + streamed

            if not streamed.strip():
                yield thinking + "\n\nEncontré información pero no pude formular una respuesta. Intenta reformular tu pregunta."

        else:
            # No tool needed — yield the response we already have
            yield assistant_text

    except Exception as exc:
        err = str(exc)
        if "429" in err or "rate" in err.lower():
            yield "Estoy recibiendo muchas solicitudes en este momento. Por favor, espera un momento e intenta de nuevo."
        elif "401" in err or "token" in err.lower():
            yield "Hay un problema de configuración con el token de API. Verifica que HF_TOKEN esté configurado correctamente."
        else:
            # Try fallback model
            try:
                fallback = call_model(messages, stream=False, model=FALLBACK_MODEL_ID)
                yield fallback.choices[0].message.content or "No pude generar una respuesta. Intenta de nuevo."
            except Exception:
                yield f"Lo siento, encontré un error. Por favor intenta de nuevo. (Error: {err[:120]})"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
EXAMPLES = [
    "Hola, ¿quién eres y qué puedes hacer?",
    "¿Cuáles son las últimas noticias sobre inteligencia artificial?",
    "Calcula la raíz cuadrada de 144 más 37 multiplicado por 2",
    "¿Cuánto es el 15% de 2450?",
    "Busca información sobre el cambio climático en 2025",
    "Calcula sin(pi/4) * cos(pi/3)",
    "¿Cuánto es 2 elevado a la potencia 10?",
    "¿Quién ganó la última Champions League?",
]

CSS = """
.gradio-container { max-width: 850px !important; margin: auto !important; }
footer { display: none !important; }
.tool-indicator { color: #888; font-style: italic; }
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
