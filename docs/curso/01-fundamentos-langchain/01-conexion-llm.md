# Módulo 1.1: Conexión con el LLM (Google Gemini)

## Objetivos
- Comprender cómo LangChain se conecta con LLMs
- Configurar Google Gemini correctamente
- Entender los parámetros del modelo
- Manejar errores y rate limits

---

## 1.1.1 ¿Qué es un LLM en LangChain?

En LangChain, un **LLM (Large Language Model)** es la interfaz básica para interactuar con modelos de lenguaje. Existen dos tipos principales:

| Tipo | Clase | Uso |
|------|-------|-----|
| **LLM** | `LLM` | Texto → Texto (modelo básico) |
| **Chat Model** | `BaseChatModel` | Mensajes → Mensaje (conversacional) |

**Para RAG y agentes, usaremos siempre `Chat Model`** porque:
- Soportan conversaciones con historial
- Manejan diferentes roles (system, user, assistant)
- Son más precisos para instrucciones complejas

---

## 1.1.2 Inicializando Google Gemini

### Código de Ejemplo

Archivo: `src/course_examples/modulo_01/01_conexion_llm.py`

```python
"""
01_conexion_llm.py
Conexión básica con Google Gemini usando LangChain

Objetivo: Entender los parámetros de configuración del modelo
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def crear_modelo_basico():
    """Crear un modelo con configuración básica"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
    )


def crear_modelo_avanzado():
    """Crear un modelo con configuración avanzada"""
    return ChatGoogleGenerativeAI(
        # Modelo: gemini-2.0-flash-exp es gratuito y rápido
        model="gemini-2.0-flash-exp",
        
        # Temperature: 0.0 (determinista) a 2.0 (creativo)
        # - 0.0-0.3: Respuestas consistentes, bueno para código/facts
        # - 0.4-0.7: Balanceado, bueno para uso general
        # - 0.8-2.0: Creativo, bueno para brainstorming
        temperature=0.5,
        
        # max_output_tokens: Límite de tokens en la respuesta
        # - None: Sin límite (usa el máximo del modelo)
        # - 100-500: Respuestas cortas
        # - 500-2000: Respuestas medias
        # - 2000+: Respuestas largas
        max_output_tokens=None,
        
        # top_p: Nucleus sampling
        # - 0.1-0.5: Más enfocado
        # - 0.6-0.9: Balanceado
        # - 0.9-1.0: Más diverso
        top_p=0.95,
        
        # timeout: Tiempo máximo de espera en segundos
        timeout=None,
        
        # max_retries: Reintentos ante fallos
        max_retries=3,
    )


def probar_modelo():
    """Probar el modelo con diferentes prompts"""
    
    llm = crear_modelo_avanzado()
    
    # Prompt 1: Pregunta factual (temperature baja sería mejor)
    print("=" * 60)
    print("PROMPT 1: Pregunta Factual")
    print("=" * 60)
    response = llm.invoke("¿Cuál es la capital de Colombia?")
    print(f"Pregunta: ¿Cuál es la capital de Colombia?")
    print(f"Respuesta: {response.content}\n")
    
    # Prompt 2: Pregunta creativa (temperature alta sería mejor)
    print("=" * 60)
    print("PROMPT 2: Pregunta Creativa")
    print("=" * 60)
    response = llm.invoke("Inventa un nombre para una startup de IA que usa animales")
    print(f"Pregunta: Inventa un nombre para una startup de IA que usa animales")
    print(f"Respuesta: {response.content}\n")
    
    # Prompt 3: Instrucción compleja
    print("=" * 60)
    print("PROMPT 3: Instrucción Compleja")
    print("=" * 60)
    prompt = """
    Eres un asistente legal experto en derecho colombiano.
    Responde de forma clara y concisa.
    
    Pregunta: ¿Qué es una acción de tutela?
    """
    response = llm.invoke(prompt)
    print(f"Respuesta: {response.content}\n")


def comparar_temperaturas():
    """Comparar respuestas con diferentes temperaturas"""
    
    print("=" * 60)
    print("COMPARACIÓN DE TEMPERATURAS")
    print("=" * 60)
    
    pregunta = "Da 3 ideas para usar IA en un bufete de abogados"
    
    for temp in [0.0, 0.5, 1.0]:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=temp,
        )
        
        print(f"\n--- Temperature: {temp} ---")
        response = llm.invoke(pregunta)
        print(f"{response.content[:200]}...\n")


def manejar_errores():
    """Demostrar manejo de errores"""
    
    from google.api_core.exceptions import ResourceExhausted
    from langchain_core.exceptions import OutputParserException
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        max_retries=3,
    )
    
    try:
        # Intentar hacer una request
        response = llm.invoke("Hola")
        print(f"✅ Success: {response.content[:50]}...")
        
    except ResourceExhausted:
        print("❌ Error: Rate limit excedido. Espera un momento.")
        
    except Exception as e:
        print(f"❌ Error inesperado: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Ejecutar pruebas
    probar_modelo()
    comparar_temperaturas()
    manejar_errores()
```

---

## 1.1.3 Parámetros del Modelo Explicados

### `model` (str)
El modelo de Gemini a usar:

| Modelo | Gratuito | Velocidad | Calidad | Uso Recomendado |
|--------|----------|-----------|---------|-----------------|
| `gemini-2.0-flash-exp` | ✅ Sí | ⚡⚡⚡ Muy rápido | Buena | **Recomendado para este curso** |
| `gemini-1.5-flash` | ✅ Sí | ⚡⚡ Rápido | Buena | Alternativa |
| `gemini-1.5-pro` | ❌ Pago | ⚡ Medio | Excelente | Tareas complejas |
| `gemini-2.0-flash-thinking_exp` | ❌ Pago | ⚡ Medio | Excelente | Razonamiento complejo |

### `temperature` (float: 0.0 - 2.0)
Controla la aleatoriedad/creatividad:

```
0.0 ──────────────────────────────────── 2.0
│         │         │         │         │
Determinista    Balance     Creativo   Caótico

Ejemplos de uso:
- 0.0-0.3: Código, matemáticas, facts
- 0.4-0.7: Chatbots, RAG, asistentes (RECOMENDADO)
- 0.8-1.5: Brainstorming, escritura creativa
- 1.5-2.0: Experimentación (poco práctico)
```

### `max_output_tokens` (int o None)
Límite de tokens en la respuesta:

```python
# Respuestas cortas (tweets, títulos)
max_output_tokens=50

# Respuestas medias (párrafos)
max_output_tokens=500

# Respuestas largas (ensayos, código)
max_output_tokens=2000

# Sin límite (usa el máximo del modelo)
max_output_tokens=None
```

### `top_p` (float: 0.0 - 1.0)
Nucleus sampling - controla la diversidad:

```python
# Más enfocado (solo top opciones)
top_p=0.5

# Balanceado (RECOMENDADO)
top_p=0.95

# Más diverso (más opciones consideradas)
top_p=1.0
```

### `timeout` (int o None)
Tiempo máximo de espera en segundos:

```python
# Sin timeout (espera indefinida)
timeout=None

# Timeout de 30 segundos
timeout=30

# Para respuestas largas
timeout=120
```

### `max_retries` (int)
Reintentos ante fallos:

```python
# Sin reintentos
max_retries=0

# 3 reintentos (RECOMENDADO)
max_retries=3

# Para producción crítica
max_retries=5
```

---

## 1.1.4 Mejores Prácticas

### ✅ DO (Haz esto)

```python
# 1. Siempre carga variables de entorno
from dotenv import load_dotenv
load_dotenv()

# 2. Usa max_retries para producción
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    max_retries=3,
)

# 3. Ajusta temperature según el caso de uso
llm_factual = ChatGoogleGenerativeAI(temperature=0.2)  # Facts
llm_creative = ChatGoogleGenerativeAI(temperature=0.8)  # Creativo

# 4. Maneja errores explícitamente
try:
    response = llm.invoke(prompt)
except Exception as e:
    logger.error(f"Error: {e}")
    response = "Lo siento, ocurrió un error."
```

### ❌ DON'T (No hagas esto)

```python
# 1. NUNCA hardcodees tu API key
llm = ChatGoogleGenerativeAI(
    google_api_key="AIzaSy..."  # ❌ MAL!
)

# 2. No uses temperature alta para facts
llm = ChatGoogleGenerativeAI(temperature=1.5)  # ❌ Para RAG

# 3. No ignores errores
response = llm.invoke(prompt)  # ❌ Sin try/except

# 4. No uses el modelo pago sin necesidad
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  # ❌ Costoso
```

---

## 1.1.5 Ejercicio Práctico

### Ejercicio 1: Configurar y Probar

1. Crea un modelo con temperature=0.3
2. Haz 3 preguntas factuales
3. Compara las respuestas con temperature=1.0

### Ejercicio 2: Manejo de Errores

Crea una función que:
- Reintente 3 veces antes de fallar
- Espere 2 segundos entre reintentos
- Retorne un mensaje amigable si falla

### Ejercicio 3: Comparación de Modelos

Compara `gemini-2.0-flash-exp` vs `gemini-1.5-flash`:
- Mide tiempo de respuesta
- Compara calidad de respuestas
- Anota diferencias

---

## 1.1.6 Recursos Adicionales

### Documentación Oficial
- [LangChain ChatGoogleGenerativeAI](https://docs.langchain.com/oss/python/langchain/integrations/google_genai/chat)
- [Google Gemini Models](https://ai.google.dev/gemini-api/docs/models/gemini)
- [LangChain LLM Interface](https://docs.langchain.com/oss/python/langchain/models/chat)

### Siguiente Lección
➡️ **1.2 Prompts: Entrada y Estructura**

---

*Lección creada: 2026-03-29*
