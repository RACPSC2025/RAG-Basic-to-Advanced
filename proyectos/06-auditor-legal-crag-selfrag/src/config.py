import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración central del Auditor Legal (Sincronizada con los modelos activos de tu API)."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # --- CONFIGURACIÓN DE MODELOS ---
    # Gemini 2.5 Flash para razonamiento (desde .env)
    LLM_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash") 
    
    # Modelo de Embeddings verificado: gemini-embedding-001
    EMBEDDING_MODEL = "models/gemini-embedding-001" 
    
    # --- CONFIGURACIÓN CHROMADB ---
    CHROMA_PATH = "./chroma_db"
    COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME", "auditoria_legal_colombiana")
    TOP_K_DOCS = 4
    
    # --- LÍMITES DEL GRAFO ---
    MAX_RETRIES = 3
    
    # --- CREDENCIALES LLAMA CLOUD ---
    LLAMA_PARSE_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
