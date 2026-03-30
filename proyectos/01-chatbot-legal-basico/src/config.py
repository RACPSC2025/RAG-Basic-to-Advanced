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

MEMORY_SHORT_TERM_K = int(os.getenv("MEMORY_SHORT_TERM_K", "5"))
MEMORY_LONG_TERM_ENABLED = os.getenv("MEMORY_LONG_TERM_ENABLED", "true").lower() == "true"

# ===========================================
# Configuración de Human in the Loop
# ===========================================

HITL_ENABLED = os.getenv("HITL_ENABLED", "true").lower() == "true"
HITL_CONFIDENCE_THRESHOLD = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.7"))

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
