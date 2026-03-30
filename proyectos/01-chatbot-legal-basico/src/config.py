"""
Configuración centralizada del Chatbot Legal.

Este módulo gestiona toda la configuración de la aplicación,
incluyendo variables de entorno y constantes.

Usa AWS Bedrock como proveedor de LLM (Amazon Nova Lite).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde la raíz del proyecto
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

# ===========================================
# Configuración de la Aplicación
# ===========================================

APP_NAME = os.getenv("APP_NAME", "Chatbot Legal Básico")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ===========================================
# Configuración de AWS Bedrock
# ===========================================

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Modelo principal para chat (Amazon Nova Lite - inference profile)
LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID",
    "arn:aws:bedrock:us-east-2:762233737662:inference-profile/us.amazon.nova-lite-v1:0"
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "amazon")  # Requerido cuando model_id es un ARN

# Parámetros del LLM
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

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
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID no está configurada en el archivo .env. "
            "Por favor configura tus credenciales de AWS."
        )

    if not aws_secret_key:
        raise ValueError(
            "AWS_SECRET_ACCESS_KEY no está configurada en el archivo .env. "
            "Por favor configura tus credenciales de AWS."
        )

    # Validar que no sean valores por defecto
    if aws_access_key == "tu_aws_access_key":
        raise ValueError(
            "AWS_ACCESS_KEY_ID tiene el valor por defecto. "
            "Por favor configura tus credenciales reales de AWS."
        )

    return True

# Validar configuración al importar
validate_config()
