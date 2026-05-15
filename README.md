# 🤖 Aria — Agente Conversacional con IA

Aplicación de agente conversacional avanzado construida con Gradio y modelos de HuggingFace. Aria tiene memoria de conversación, puede buscar información en la web y realizar cálculos matemáticos complejos.

**[🚀 Ver app desplegada en HuggingFace Spaces](https://huggingface.co/spaces/moisesalejandro/aria-assistant)**

## Características

- **Memoria conversacional** — Recuerda el historial completo de la conversación
- **Búsqueda web** — Consulta DuckDuckGo para información actualizada (sin API key)
- **Calculadora segura** — Evalúa expresiones matemáticas via AST (aritmética, trigonometría, logaritmos)
- **Respuestas en streaming** — Retroalimentación en tiempo real
- **Manejo de errores** — Modelo de fallback, mensajes claros ante fallos de API
- **Interfaz bilingüe** — UI y respuestas en español

## Arquitectura

```
Usuario → Gradio ChatInterface
              ↓
          chat() — construye historial + mensajes
              ↓
          HuggingFace InferenceClient
          (Qwen2.5-72B-Instruct)
              ↓
     ¿Respuesta contiene <tool_call>?
          /              \
        Sí               No
         ↓                ↓
   Ejecutar herramienta  Mostrar respuesta
   (web_search / calc)
         ↓
   Llamada final al modelo (streaming)
         ↓
   Respuesta con información de la herramienta
```

**Decisiones de diseño:**
- **Llamadas directas a la API** (sin LangChain) — Menos dependencias, código más transparente
- **Tool calling con marcadores XML** (`<tool_call>`) — Más fiable que el parámetro `tools` nativo entre modelos gratuitos
- **AST para la calculadora** — Evita `eval()` inseguro, demuestra buenas prácticas de seguridad
- **Primera pasada no-streaming** — Permite detectar llamadas a herramientas antes de mostrar la respuesta

## Tecnologías

| Componente | Tecnología |
|------------|------------|
| Interfaz | Gradio 5.x ChatInterface |
| LLM principal | Qwen/Qwen2.5-72B-Instruct |
| LLM fallback | meta-llama/Llama-3.3-70B-Instruct |
| Cliente API | huggingface_hub InferenceClient |
| Búsqueda web | duckduckgo-search |
| Calculadora | Módulo `ast` de Python (evaluación segura) |
| Despliegue | HuggingFace Spaces |

## Instalación local

### Requisitos
- Python 3.10+
- Cuenta en HuggingFace con API token

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/moisesrodriguez/aria-assistant.git
cd aria-assistant

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar token de HuggingFace
export HF_TOKEN=tu_token_aqui
# En Windows: set HF_TOKEN=tu_token_aqui

# 5. Ejecutar la app
python app.py
```

La app estará disponible en `http://localhost:7860`

### Obtener un HuggingFace API Token
1. Crea una cuenta en [huggingface.co](https://huggingface.co)
2. Ve a Settings → Access Tokens
3. Crea un nuevo token con permisos de lectura

## Despliegue en HuggingFace Spaces

1. Crea un nuevo Space en [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Visibilidad: **Public**
2. En Settings → Repository Secrets, añade: `HF_TOKEN` = tu token
3. Sube los archivos `app.py`, `requirements.txt`, `README.md`
4. El Space se construye y despliega automáticamente

## Ejemplos de uso

| Tipo | Ejemplo |
|------|---------|
| Chat básico | "¿Quién eres y qué puedes hacer?" |
| Búsqueda web | "¿Cuáles son las últimas noticias sobre IA?" |
| Calculadora | "Calcula la raíz cuadrada de 2 elevado a 16" |
| Memoria | Pregunta algo, luego pregunta "¿Qué te acabo de preguntar?" |
| Matemáticas avanzadas | "Calcula sin(pi/4) * cos(pi/3)" |

## Estructura del proyecto

```
aria-assistant/
├── app.py              # Aplicación principal (toda la lógica)
├── requirements.txt    # Dependencias Python
└── README.md           # Este archivo
```

## Autor

Moises Rodriguez — Máster en Inteligencia Artificial, Módulo Python para IA
