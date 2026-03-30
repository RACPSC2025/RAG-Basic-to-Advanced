import logging
import os
import time
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from .state import AgentState
from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODELOS DE SALIDA ESTRUCTURADA ---
class DocumentGrade(BaseModel):
    """Calificación binaria para un documento."""
    relevance: str = Field(description="¿Es relevante? 'yes' o 'no'")

class GradeBatch(BaseModel):
    """Calificación de un lote de documentos para ahorrar peticiones."""
    grades: List[DocumentGrade] = Field(description="Lista de calificaciones para los documentos proporcionados")

class GradeHallucination(BaseModel):
    """Calificación de alucinación."""
    binary_score: str = Field(description="¿La respuesta es fiel a los documentos? 'yes' o 'no'")

class GradeAnswer(BaseModel):
    """Calificación de utilidad."""
    binary_score: str = Field(description="¿Resuelve la pregunta? 'yes' o 'no'")

class AuditorNodes:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=0)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
        # --- DEFINICIÓN DE PROMPTS Y CADENAS ---
        # 1. Calificador por Lotes
        self.doc_grader = ChatPromptTemplate.from_template(
            """Evalúa la relevancia de los siguientes {num_docs} documentos para la pregunta legal.
            Pregunta: {question}
            Documentos: {documents_text}
            Responde estrictamente con una lista de 'yes' o 'no' para cada uno."""
        ) | self.llm.with_structured_output(GradeBatch)

        # 2. Otros calificadores
        self.hallucination_grader = ChatPromptTemplate.from_template(
            "Analiza: {documents} vs {generation}. ¿Es fiel? 'yes'/'no'"
        ) | self.llm.with_structured_output(GradeHallucination)

        self.answer_grader = ChatPromptTemplate.from_template(
            "Pregunta: {question}\nRespuesta: {generation}. ¿Resuelve la duda? 'yes'/'no'"
        ) | self.llm.with_structured_output(GradeAnswer)

        # Vector Store
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PATH
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.TOP_K_DOCS})

    def _wait_for_rate_limit(self):
        """Pausa dinámica basada en el RPM configurado en .env."""
        if Config.REQUEST_DELAY > 0:
            time.sleep(Config.REQUEST_DELAY)

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"--- RECUPERANDO PARA: {state['question']} ---")
        docs = self.retriever.invoke(state["question"])
        self._wait_for_rate_limit()
        return {"documents": docs, "steps": state["steps"] + ["retrieve"]}

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- CALIFICANDO LOTE DE DOCUMENTOS ---")
        question = state["question"]
        documents = state["documents"]
        
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"\n--- DOC {i+1} ---\n{doc.page_content}\n"
        
        batch_results = self.doc_grader.invoke({
            "num_docs": len(documents),
            "question": question,
            "documents_text": docs_text
        })
        
        filtered_docs = []
        is_relevant = "no"
        for i, result in enumerate(batch_results.grades):
            if result.relevance == "yes":
                filtered_docs.append(documents[i])
                is_relevant = "yes"
        
        self._wait_for_rate_limit()
        return {"documents": filtered_docs, "is_relevant": is_relevant, "steps": state["steps"] + ["grade_documents"]}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- GENERANDO RESPUESTA ---")
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        prompt = f"Contexto Legal:\n{context}\n\nPregunta: {state['question']}\nRespuesta profesional:"
        generation = self.llm.invoke(prompt).content
        self._wait_for_rate_limit()
        return {"generation": generation, "steps": state["steps"] + ["generate"]}

    def transform_query(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- OPTIMIZANDO CONSULTA ---")
        prompt = f"Optimiza esta consulta legal para búsqueda en Colombia: {state['question']}"
        better_query = self.llm.invoke(prompt).content
        self._wait_for_rate_limit()
        return {"question": better_query, "retries": state["retries"] + 1, "steps": state["steps"] + ["transform_query"]}

    def grade_generation_v_documents(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- VALIDANDO ALUCINACIONES ---")
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        score = self.hallucination_grader.invoke({"documents": context, "generation": state["generation"]})
        self._wait_for_rate_limit()
        return {"hallucination": score.binary_score, "steps": state["steps"] + ["grade_hallucination"]}

    def grade_generation_v_question(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- VALIDANDO UTILIDAD ---")
        score = self.answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        self._wait_for_rate_limit()
        return {"answer_relevant": score.binary_score, "steps": state["steps"] + ["grade_answer"]}
