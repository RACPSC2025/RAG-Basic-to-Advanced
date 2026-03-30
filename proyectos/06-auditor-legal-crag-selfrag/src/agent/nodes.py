import logging
import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from .state import AgentState
from ..config import Config

# --- OPCIONAL: QDRANT IMPORTS (COMENTADOS) ---
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient

# Configuración de Logging profesional
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos para validación estructurada (Pydantic)
class GradeDocuments(BaseModel):
    """Calificación binaria de relevancia de documentos."""
    binary_score: str = Field(description="¿Es el documento relevante para la pregunta? 'yes' o 'no'")

class GradeHallucination(BaseModel):
    """Calificación de alucinación."""
    binary_score: str = Field(description="¿Está la respuesta basada en los documentos? 'yes' o 'no'")

class GradeAnswer(BaseModel):
    """Calificación de la utilidad de la respuesta."""
    binary_score: str = Field(description="¿Resuelve la respuesta la pregunta? 'yes' o 'no'")

# Nodos del Grafo
class AuditorNodes:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=0)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
        # LLMs con estructura de salida (Structured Output)
        self.doc_grader = self.llm.with_structured_output(GradeDocuments)
        self.hallucination_grader = self.llm.with_structured_output(GradeHallucination)
        self.answer_grader = self.llm.with_structured_output(GradeAnswer)
        
        # --- VECTOR STORE: CHROMADB (ACTIVO) ---
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PATH
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.TOP_K_DOCS})

        # --- VECTOR STORE: QDRANT CLOUD (COMENTADO) ---
        """
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=Config.COLLECTION_NAME,
            embedding=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.TOP_K_DOCS})
        """

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Recupera documentos de ChromaDB."""
        logger.info(f"--- RECUPERANDO DOCUMENTOS PARA: {state['question']} ---")
        
        # Recuperar de Chroma
        docs = self.retriever.invoke(state["question"])
        
        return {
            "documents": docs, 
            "steps": state["steps"] + ["retrieve"]
        }

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Evalúa la relevancia de los documentos recuperados (CRAG)."""
        logger.info("--- CALIFICANDO DOCUMENTOS ---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        is_relevant = "no"
        
        for doc in documents:
            score = self.doc_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            if score.binary_score == "yes":
                filtered_docs.append(doc)
                is_relevant = "yes"
        
        return {
            "documents": filtered_docs, 
            "is_relevant": is_relevant,
            "steps": state["steps"] + ["grade_documents"]
        }

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Genera respuesta legal (Self-RAG)."""
        logger.info("--- GENERANDO RESPUESTA ---")
        question = state["question"]
        documents = state["documents"]
        
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"Basado en el contexto: {context}, responde la pregunta: {question}. Si no sabes, di que no sabes."
        
        generation = self.llm.invoke(prompt).content
        return {
            "generation": generation,
            "steps": state["steps"] + ["generate"]
        }

    def transform_query(self, state: AgentState) -> Dict[str, Any]:
        """Re-escribe la consulta si los documentos no son útiles (Corrective)."""
        logger.info("--- TRANSFORMANDO CONSULTA ---")
        question = state["question"]
        prompt = f"Optimiza esta consulta legal para mejorar la búsqueda en base vectorial: {question}"
        
        better_query = self.llm.invoke(prompt).content
        return {
            "question": better_query,
            "retries": state["retries"] + 1,
            "steps": state["steps"] + ["transform_query"]
        }

    def grade_generation_v_documents(self, state: AgentState) -> Dict[str, Any]:
        """Verifica alucinaciones (Self-RAG)."""
        logger.info("--- VERIFICANDO ALUCINACIONES ---")
        generation = state["generation"]
        documents = state["documents"]
        
        context = "\n\n".join([doc.page_content for doc in documents])
        score = self.hallucination_grader.invoke(
            {"documents": context, "generation": generation}
        )
        
        return {
            "hallucination": score.binary_score,
            "steps": state["steps"] + ["grade_hallucination"]
        }

    def grade_generation_v_question(self, state: AgentState) -> Dict[str, Any]:
        """Verifica si la respuesta es útil."""
        logger.info("--- VERIFICANDO UTILIDAD ---")
        generation = state["generation"]
        question = state["question"]
        
        score = self.answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        
        return {
            "answer_relevant": score.binary_score,
            "steps": state["steps"] + ["grade_answer"]
        }
