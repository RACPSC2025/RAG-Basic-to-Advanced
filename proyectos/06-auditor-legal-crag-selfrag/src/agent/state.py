from typing import List, TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Representa el estado del grafo del Auditor Legal.
    """
    question: str               # Pregunta original del usuario
    generation: str             # Respuesta generada por el LLM
    documents: List[Document]   # Lista de documentos relevantes recuperados
    retries: int                # Contador de reintentos de búsqueda
    steps: List[str]            # Historial de pasos tomados (auditoría)
    search_query: str           # Consulta optimizada (si se transforma)
    
    # Calificaciones
    is_relevant: str            # 'yes' si los docs son útiles
    hallucination: str          # 'yes' si hay alucinación, 'no' si es fiel
    answer_relevant: str        # 'yes' si la respuesta resuelve la duda
