# 🤖 Proyecto 1: Chatbot Legal Básico

> **Nivel**: Básico (Refrescamiento)  
> **Tiempo Estimado**: 2-4 horas  
> **Tecnologías**: LangChain, Google Gemini, Memoria  
> **Estado**: ✅ En Desarrollo

---

## 📋 Índice

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Requisitos Previos](#requisitos-previos)
3. [Fase 1: Importación y Configuración](#fase-1-importación-y-configuración)
4. [Fase 2: Invocar Modelo](#fase-2-invocar-modelo)
5. [Fase 3: Chat Prompt Template](#fase-3-chat-prompt-template)
6. [Fase 4: System Prompt](#fase-4-system-prompt)
7. [Fase 5: Response + Parsing](#fase-5-response--parsing)
8. [Fase 6: Memoria Corto Plazo](#fase-6-memoria-corto-plazo)
9. [Fase 7: Memoria Largo Plazo](#fase-7-memoria-largo-plazo)
10. [Fase 8: Human in the Loop](#fase-8-human-in-the-loop)
11. [Fase 9: Testing](#fase-9-testing)
12. [Fase 10: Empaquetado](#fase-10-empaquetado)

---

## Descripción del Proyecto

Chatbot conversacional para consultas legales básicas sin retrieval de documentos. Usa únicamente el conocimiento del LLM con memoria de conversación.

**Objetivos de Aprendizaje**:
- ✅ Repasar fundamentos de LangChain
- ✅ Configurar Google Gemini API
- ✅ Implementar memoria de conversación
- ✅ Human in the Loop para aprobación
- ✅ Testing básico de agentes

**Resultado Final**: Chatbot funcional que responde consultas legales básicas manteniendo contexto de conversación y pidiendo aprobación para respuestas críticas.

---

## Requisitos Previos

### Técnicos

```bash
# Python 3.12+
python --version  # Debe ser >= 3.12

# UV package manager (recomendado)
uv --version

# O pip tradicional
pip --version
```

### Conocimientos

- ✅ Módulos 1-3 completados (Fundamentos, Memoria, Streaming)
- ✅ Conceptos básicos de LangChain
- ✅ Python intermedio

### Configuración

```bash
# 1. Clonar o navegar al directorio del proyecto
cd proyectos/01-chatbot-legal-basico

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Fase 1: Importación y Configuración

### Objetivo

Configurar el entorno del proyecto y las variables de entorno necesarias.

### Paso 1.1: Archivo `.env`

Crear archivo `.env` en la raíz del proyecto:

```bash
# .env
# Google Gemini API Key
GOOGLE_API_KEY=tu_api_key_aqui

# Configuración de la aplicación
APP_NAME=Chatbot Legal Básico
APP_VERSION=1.0.0
LOG_LEVEL=INFO

# Configuración del LLM
LLM_MODEL=gemini-2.0-flash-exp
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000
```

### Paso 1.2: Archivo `requirements.txt`

```txt
# requirements.txt
# Core LangChain
langchain>=0.3.0
langchain-core>=0.3.0
langchain-google-genai>=2.0.0

# Utilidades
python-dotenv>=1.0.0
pydantic>=2.0.0

# Testing
pytest>=8.0.0
pytest-cov>=5.0.0

# Logging (ya incluido en Python stdlib)
# typing (ya incluido en Python 3.12+)
```

### Paso 1.3: Estructura de Directorios

```
01-chatbot-legal-basico/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuración centralizada
│   ├── chatbot.py          # Lógica principal del chatbot
│   └── memory.py           # Gestión de memoria
├── tests/
│   ├── __init__.py
│   ├── test_chatbot.py     # Tests del chatbot
│   └── test_memory.py      # Tests de memoria
├── docs/
│   └── README.md           # Este archivo
├── .env                    # Variables de entorno (NO subir a Git)
├── .env.example            # Ejemplo de .env (SÍ subir a Git)
├── requirements.txt        # Dependencias
└── main.py                 # Punto de entrada
```

### Paso 1.4: Módulo de Configuración

```python
# src/config.py
"""
Configuración centralizada del Chatbot Legal.

Este módulo gestiona toda la configuración de la aplicación,
incluyendo variables de entorno y constantes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ===========================================
# Configuración de la Aplicación
# ===========================================

APP_NAME = os.getenv("APP_NAME", "Chatbot Legal Básico")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ===========================================
# Configuración del LLM
# ===========================================

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

# ===========================================
# Configuración de Memoria
# ===========================================

MEMORY_SHORT_TERM_K = 5  # Últimos 5 turnos para memoria corto plazo
MEMORY_LONG_TERM_ENABLED = True  # Habilitar resumen de largo plazo

# ===========================================
# Configuración de Human in the Loop
# ===========================================

HITL_ENABLED = True  # Habilitar aprobación humana
HITL_CONFIDENCE_THRESHOLD = 0.7  # Threshold para aprobación automática

# ===========================================
# Paths
# ===========================================

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Crear directorios si no existen
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ===========================================
# Validación de Variables Críticas
# ===========================================

def validate_config():
    """
    Validar que las variables de entorno críticas estén configuradas.
    
    Raises:
        ValueError: Si alguna variable crítica falta
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY no está configurada en el archivo .env. "
            "Por favor obtén una API Key en https://aistudio.google.com/app/apikey"
        )
    
    if google_api_key == "tu_api_key_aqui":
        raise ValueError(
            "GOOGLE_API_KEY tiene el valor por defecto. "
            "Por favor configura tu API Key real en el archivo .env"
        )
    
    return True

# Validar configuración al importar
validate_config()
```

---

## Fase 2: Invocar Modelo

### Objetivo

Configurar e inicializar el modelo Google Gemini con LangChain.

### Paso 2.1: Módulo de Inicialización del LLM

```python
# src/llm.py
"""
Módulo para inicialización y configuración del LLM.

Proporciona funciones para crear y configurar el modelo
Google Gemini con los parámetros adecuados.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from .config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LOG_LEVEL
)
import logging

# Configurar logging
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


def create_llm(
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    streaming: bool = True,
    max_retries: int = 3
) -> BaseChatModel:
    """
    Crear instancia del LLM Google Gemini.
    
    Args:
        model: Nombre del modelo (default: de config)
        temperature: Temperatura del modelo (0.0-2.0)
        max_tokens: Máximo de tokens en la respuesta
        streaming: Si habilitar streaming de tokens
        max_retries: Máximo de reintentos ante fallos
    
    Returns:
        Instancia configurada de ChatGoogleGenerativeAI
    
    Raises:
        ValueError: Si los parámetros están fuera de rango
    """
    
    # Usar valores por defecto de config si no se proporcionan
    model = model or LLM_MODEL
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens or LLM_MAX_TOKENS
    
    # Validar parámetros
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"temperature debe estar entre 0.0 y 2.0, got {temperature}")
    
    if max_tokens <= 0:
        raise ValueError(f"max_tokens debe ser positivo, got {max_tokens}")
    
    logger.info(f"Inicializando LLM: {model}")
    logger.debug(f"Parámetros: temperature={temperature}, max_tokens={max_tokens}")
    
    try:
        # Crear instancia del LLM
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            streaming=streaming,
            max_retries=max_retries,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        logger.info(f"LLM inicializado exitosamente: {model}")
        
        # Test rápido de conexión
        test_response = llm.invoke("Hola, responde solo con 'OK'")
        logger.debug(f"Test de conexión: {test_response.content[:20]}...")
        
        return llm
        
    except Exception as e:
        logger.error(f"Error inicializando LLM: {e}")
        raise


def get_llm_for_task(task_complexity: str) -> BaseChatModel:
    """
    Obtener el LLM adecuado según la complejidad de la tarea.
    
    Args:
        task_complexity: Nivel de complejidad ('simple', 'medium', 'complex')
    
    Returns:
        Instancia de LLM configurada apropiadamente
    """
    
    if task_complexity == "simple":
        # Tareas simples: modelo más rápido/económico
        return create_llm(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_tokens=500
        )
    elif task_complexity == "medium":
        # Tareas medias: balance
        return create_llm(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            max_tokens=1000
        )
    else:  # complex
        # Tareas complejas: modelo más capaz
        return create_llm(
            model="gemini-2.0-flash-exp",
            temperature=0.5,
            max_tokens=2000
        )


# LLM por defecto para la aplicación
default_llm = None


def get_default_llm() -> BaseChatModel:
    """
    Obtener o crear el LLM por defecto de la aplicación.
    
    Returns:
        Instancia de LLM configurada
    """
    global default_llm
    
    if default_llm is None:
        default_llm = create_llm()
    
    return default_llm
```

---

*(Continuaré con las fases restantes en el siguiente archivo debido a la extensión...)*
