import logging
from langchain_chroma import Chroma
from proyectos.Rag_Legal.config import settings, get_embeddings

logger = logging.getLogger(__name__)

def get_vector_store() -> Chroma:
    """
    Se conecta a la base de datos Chroma existente en la carpeta storage.
    Asume que los embeddings fueron creados usando 'models/embedding-001' de Google.
    """
    embeddings = get_embeddings()
    # Conectamos al directorio SQLite especificado en el storage
    vector_store = Chroma(
        persist_directory=settings.STORAGE_PATH,
        embedding_function=embeddings
    )
    return vector_store

def get_strict_retriever(k: int = 4):
    """
    Devuelve un retriever estricto configurado para búsqueda legal.
    En versiones avanzadas se puede agregar un threshold de score similarity_score_threshold.
    """
    vector_store = get_vector_store()
    
    # Búsqueda basada en similitud.
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
