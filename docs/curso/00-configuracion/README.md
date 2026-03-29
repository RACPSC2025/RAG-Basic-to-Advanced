# Módulo 0: Configuración del Entorno

## Objetivos
- Configurar el entorno de desarrollo
- Instalar todas las dependencias necesarias
- Configurar variables de entorno
- Verificar que todo funcione correctamente

---

## 0.1 Instalación de Dependencias

### Paso 1: Verificar Python

Este curso requiere **Python 3.12 o superior**.

```bash
python --version
# Debería mostrar: Python 3.12.x
```

### Paso 2: Instalar Dependencias

El proyecto usa **UV** como package manager (más rápido que pip).

```bash
# Instalar dependencias con UV
uv sync

# O si prefieres pip tradicional
pip install -r requirements.txt
```

### Dependencias Clave

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `langchain` | >=1.2.13 | Framework principal |
| `langchain-google-genai` | >=4.2.1 | Integración con Gemini |
| `langgraph` | >=1.1.3 | Orquestación de agentes |
| `google-generativeai` | >=0.8.6 | SDK de Google Gemini |
| `python-dotenv` | >=1.2.2 | Variables de entorno |
| `qdrant-client` | >=1.17.1 | Vector store |

---

## 0.2 Configuración de Variables de Entorno

### Paso 1: Crear archivo `.env`

Crea un archivo llamado `.env` en la raíz del proyecto:

```bash
# .env
# Google Gemini API Key
GOOGLE_API_KEY=tu_api_key_aqui

# LlamaParse API Key (opcional, para parsing avanzado)
LLAMA_PARSE_API_KEY=tu_api_key_aqui

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Logs
LOGS_PATH=./logs
```

### Paso 2: Obtener tu API Key de Google Gemini

1. Ve a [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Inicia sesión con tu cuenta de Google
3. Haz clic en "Create API Key"
4. Copia la clave y pégala en tu `.env`

**Nota**: La API gratuita de Gemini tiene los siguientes límites:
- 15 requests por minuto (RPM)
- 1 millón de tokens por minuto (TPM)
- 1,500 requests por día (RPD)

### Paso 3: Verificar Variables de Entorno

Crea un script de prueba:

```python
# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")

if google_key and google_key != "tu_api_key_aqui":
    print("✅ GOOGLE_API_KEY configurada correctamente")
else:
    print("❌ GOOGLE_API_KEY no configurada o inválida")
```

Ejecuta:
```bash
python test_env.py
```

---

## 0.3 Estructura del Proyecto

La estructura recomendada para este curso es:

```
RAG MVP/
├── .env                           # Variables de entorno (NO subir a Git)
├── .gitignore                     # Ignorar .env y archivos temporales
├── requirements.txt               # Dependencias
├── pyproject.toml                 # Configuración del proyecto
│
├── docs/
│   └── curso/                     # Documentación del curso
│       ├── 00-configuracion/
│       ├── 01-fundamentos-langchain/
│       └── ...
│
├── src/
│   ├── course_examples/           # Código de ejemplo del curso
│   │   ├── modulo_01/
│   │   ├── modulo_02/
│   │   └── ...
│   ├── agent/                     # Tu código de agentes
│   ├── indexing/                  # Tu código de indexación
│   ├── ingestion/                 # Tu código de ingestión
│   ├── retrieval/                 # Tu código de retrieval
│   ├── tools/                     # Tus herramientas
│   └── utils/                     # Utilidades
│
├── data/
│   ├── input/                     # Documentos de entrada
│   └── processed/                 # Documentos procesados
│
├── storage/                       # Almacenamiento vectorial
├── logs/                          # Logs de ejecución
└── test/                          # Tests unitarios
```

---

## 0.4 Primeros Pasos con LangChain

### Script de Verificación

Crea el archivo `src/course_examples/modulo_01/00_hello_langchain.py`:

```python
"""
00_hello_langchain.py
Tu primer script con LangChain y Google Gemini

Objetivo: Verificar que la configuración funciona correctamente
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar variables de entorno
load_dotenv()

def main():
    """Función principal de prueba"""
    
    # 1. Verificar API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        print("❌ Error: GOOGLE_API_KEY no configurada en el archivo .env")
        print("   Sigue las instrucciones en docs/curso/00-configuracion/README.md")
        return
    
    print("✅ API Key encontrada")
    
    # 2. Inicializar el modelo
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Modelo gratuito recomendado
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    print(f"✅ Modelo inicializado: {llm.model}")
    
    # 3. Hacer una pregunta simple
    response = llm.invoke("Hola, ¿cómo estás? Responde brevemente.")
    
    print(f"\n🤖 Gemini dice: {response.content}")
    print("\n✅ ¡Configuración exitosa! LangChain + Gemini funcionan correctamente.")

if __name__ == "__main__":
    main()
```

### Ejecutar Prueba

```bash
python src/course_examples/modulo_01/00_hello_langchain.py
```

**Salida esperada:**
```
✅ API Key encontrada
✅ Modelo inicializado: gemini-2.0-flash-exp

🤖 Gemini dice: ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú?

✅ ¡Configuración exitosa! LangChain + Gemini funcionan correctamente.
```

---

## 0.5 Solución de Problemas

### Error: "GOOGLE_API_KEY no configurada"

**Causa**: El archivo `.env` no existe o no está en la ruta correcta.

**Solución**:
1. Verifica que `.env` esté en la raíz del proyecto
2. Asegúrate de que la línea sea exactamente: `GOOGLE_API_KEY=tu_clave`
3. Reinicia tu terminal/IDE después de crear el `.env`

### Error: "403 Forbidden" o "API_KEY_INVALID"

**Causa**: La API Key es inválida o expiró.

**Solución**:
1. Ve a [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Genera una nueva API Key
3. Actualiza tu archivo `.env`

### Error: "Rate limit exceeded"

**Causa**: Excediste el límite de requests de la API gratuita.

**Solución**:
1. Espera un minuto y reintenta
2. Los límites son: 15 RPM, 1M TPM, 1500 RPD
3. Considera upgrade a la API de pago si necesitas más

### Error: ModuleNotFoundError

**Causa**: Las dependencias no están instaladas.

**Solución**:
```bash
uv sync
# o
pip install -r requirements.txt
```

---

## 0.6 Recursos Adicionales

### Documentación Oficial
- [LangChain Installation](https://docs.langchain.com/oss/python/langchain/installation)
- [Google Gemini API](https://ai.google.dev/docs)
- [LangChain Google Integration](https://docs.langchain.com/oss/python/langchain/integrations/google)

### Próximos Pasos
Una vez completada la configuración, continúa con:
- **Lección 1.1**: Conexión con el LLM (Google Gemini)
- **Lección 1.2**: Prompts - Entrada y Estructura

---

## Ejercicio Práctico

1. ✅ Configura tu archivo `.env` con la API Key de Google
2. ✅ Ejecuta el script `00_hello_langchain.py`
3. ✅ Modifica el script para hacer 3 preguntas diferentes a Gemini
4. ✅ Experimenta cambiando el parámetro `temperature` (0.0 a 1.0)

### Ejemplo de Ejercicio Completado

```python
# Mi versión modificada de 00_hello_langchain.py

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,  # Más determinista
)

preguntas = [
    "¿Cuál es la capital de Francia?",
    "Explica qué es un agujero negro en una frase",
    "¿Cuánto es 2 + 2?"
]

for i, pregunta in enumerate(preguntas, 1):
    response = llm.invoke(pregunta)
    print(f"{i}. P: {pregunta}")
    print(f"   R: {response.content}\n")
```

---

*Lección creada: 2026-03-29*
