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
MAX_TOKENS = 512

client = InferenceClient(token=HF_TOKEN)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Aria, a helpful AI assistant. You speak the same language as the user.

You have two tools available. Use them when needed:

TOOL 1 - web_search: Use for current events, news, recent facts, sports results.
Format: <tool_call>{"name": "web_search", "arguments": {"query": "search query here"}}</tool_call>

TOOL 2 - calculator: Use for math calculations, square roots, trigonometry, etc.
Format: <tool_call>{"name": "calculator", "arguments": {"expression": "math expression here"}}</tool_call>

Rules:
- If you need a tool, output ONLY the <tool_call> tag. Nothing else.
- If you don't need a tool, respond directly and naturally.
- Never use tool_call for questions you can answer from memory.

Examples of when to use tools:
- "Who won yesterday's match?" → use web_search
- "What is sqrt(64)?" → use calculator
- "What is 5+3?" → answer directly: "8"
- "Hello, who are you?" → answer directly
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
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e,
}


def safe_eval_math(expression: str) -> str:
    try:
        expression = expression.strip().replace("^", "**")
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return "Error: operación no permitida."
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
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No se encontraron resultados."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Sin título")
            body = r.get("body", "")[:300]
            href = r.get("href", "")
            lines.append(f"**{i}. {title}**\n{body}\n[Fuente]({href})")
        return "\n\n---\n\n".join(lines)
    except Exception as exc:
        return f"Error en búsqueda: {exc}"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def execute_tool(name: str, arguments: dict) -> str:
    if name == "web_search":
        return web_search(arguments.get("query", "").strip())
    if name == "calculator":
        return safe_eval_math(arguments.get("expression", "").strip())
    return f"Error: herramienta '{name}' desconocida."


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_TOOL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> tuple:
    match = _TOOL_PATTERN.search(text)
    if not match:
        return None, text
    try:
        tc = json.loads(match.group(1).strip())
        if "name" in tc and "arguments" in tc:
            return tc, text[: match.start()].strip()
    except json.JSONDecodeError:
        pass
    return None, text


def extract_native_tool_call(response) -> dict | None:
    try:
        tcs = response.choices[0].message.tool_calls
        if tcs:
            tc = tcs[0]
            return {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
    except Exception:
        pass
    return None


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# Build messages
# ---------------------------------------------------------------------------
def build_messages(history: list, user_message: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        text = extract_text(msg.get("content", ""))
        if text:
            msgs.append({"role": msg["role"], "content": text})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def call_model(messages: list, stream: bool = False, model: str = MODEL_ID,
               max_tokens: int = MAX_TOKENS):
    return client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.6,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def chat(message: str, history: list):
    messages = build_messages(history, message)

    # Step 1 — call model to get response or tool decision
    try:
        response = call_model(messages, stream=False, max_tokens=300)
        msg_obj = response.choices[0].message
        assistant_text = (msg_obj.content or "").strip()

        if not assistant_text:
            native = extract_native_tool_call(response)
            if native:
                assistant_text = f'<tool_call>{json.dumps(native)}</tool_call>'
            else:
                # Empty first response — try fallback model
                try:
                    fb = call_model(messages, stream=False, model=FALLBACK_MODEL_ID, max_tokens=300)
                    assistant_text = (fb.choices[0].message.content or "").strip()
                except Exception:
                    pass
                if not assistant_text:
                    yield "⚠️ El modelo no generó respuesta. Por favor intenta de nuevo."
                    return

    except Exception as exc:
        err = str(exc)
        if "429" in err or "rate" in err.lower():
            yield "⏳ Demasiadas solicitudes. Espera unos segundos e intenta de nuevo."
        else:
            yield f"❌ Error al contactar el modelo: {err[:200]}"
        return

    # Step 2 — check for tool call
    tool_call, text_before = parse_tool_call(assistant_text)

    if not tool_call:
        # No tool needed: stream a full quality response
        try:
            streamed = ""
            for chunk in call_model(messages, stream=True, max_tokens=MAX_TOKENS):
                delta = chunk.choices[0].delta.content or ""
                streamed += delta
                yield streamed
            if not streamed.strip():
                yield assistant_text or "No pude generar una respuesta."
        except Exception:
            yield assistant_text or "No pude generar una respuesta."
        return

    # Step 3 — execute tool and format result directly (no second model call)
    tool_name = tool_call["name"]
    tool_args = tool_call["arguments"]

    yield f"🔧 *Consultando {tool_name}...*"

    tool_result = execute_tool(tool_name, tool_args)

    if tool_name == "calculator":
        expr = tool_args.get("expression", "")
        display_expr = expr.replace("**", "^")
        yield f"🧮 **Calculadora**\n\n`{display_expr}` = **{tool_result}**"

    elif tool_name == "web_search":
        query = tool_args.get("query", "")
        yield f"🔍 **Búsqueda:** *{query}*\n\n{tool_result}"

    else:
        yield f"**Resultado ({tool_name}):**\n\n{tool_result}"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["Hola, ¿quién eres y qué puedes hacer?"],
    ["¿Cuáles son las últimas noticias sobre inteligencia artificial?"],
    ["Calcula la raíz cuadrada de 144"],
    ["¿Cuánto es el 15% de 2450?"],
    ["¿Quién ganó la última Champions League?"],
    ["Calcula sin(pi/4) * cos(pi/3)"],
    ["Busca información sobre el cambio climático"],
    ["¿Cuánto es 2 elevado a la potencia 10?"],
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
