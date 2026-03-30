import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración central desacoplada del Auditor Legal."""
    
    # --- CREDENCIALES ---
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLAMA_PARSE_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    
    # --- MODELOS ---
    LLM_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash") 
    EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001") 
    
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
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
