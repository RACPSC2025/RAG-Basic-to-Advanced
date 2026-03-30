import os
from dotenv import load_dotenv

# Cargar variables de entorno desde la raíz del proyecto
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_ROOT / ".env")

class Config:
    """Configuración central desacoplada del Auditor Legal.
    
    Migrado a AWS Bedrock - 2026-03-30
    """

    # --- CREDENCIALES AWS ---
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
    
    # --- CREDENCIALES ADICIONALES ---
    LLAMA_PARSE_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

    # --- MODELOS (AWS Bedrock) ---
    # LLM: Amazon Nova Lite (inference profile)
    LLM_MODEL_ID = os.getenv(
        "LLM_MODEL_ID",
        "arn:aws:bedrock:us-east-2:762233737662:inference-profile/us.amazon.nova-lite-v1:0"
    )
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "amazon")
    
    # Embeddings: Amazon Titan Text v2
    EMBEDDING_MODEL_ID = os.getenv(
        "EMBEDDING_MODEL_ID",
        "amazon.titan-embed-text-v2:0"
    )

    # --- VECTOR STORE (CHROMADB) ---
    CHROMA_PATH = os.getenv("DATA_STORAGE_PATH", "./storage/")
    COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME", "auditoria_legal_colombiana")
    TOP_K_DOCS = int(os.getenv("TOP_K_DOCS", 4))

    # --- 🚀 GESTIÓN DINÁMICA DE RATE LIMITS & BATCHING ---
    # RPM (Requests Per Minute)
    RPM = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", 15))
    # Burst Limit (Cuántas peticiones seguidas antes de forzar pausa)
    BURST = int(os.getenv("RATE_LIMIT_BURST_LIMIT", 5))

    # Cálculo automático de retraso para no exceder RPM: (60s / RPM)
    # Si RPM es 0 o muy alto, el retraso será 0.
    REQUEST_DELAY = 60.0 / RPM if RPM > 0 else 0

    # Tamaño de lote para ingesta (puede definirse en .env, default: BURST)
    INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", BURST))

    # Máximos reintentos del Grafo
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # Parámetros del LLM
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
