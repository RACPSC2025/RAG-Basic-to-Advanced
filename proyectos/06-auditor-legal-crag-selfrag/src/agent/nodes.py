"""
Nodes del Auditor Legal - Proyecto 6 (CRAG + Self-RAG)

Migrado a AWS Bedrock - 2026-03-30
"""

import logging
import time
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from .state import AgentState
from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODELOS DE SALIDA ESTRUCTURADA ---
class GradeSingle(BaseModel):
    """Calificación binaria para un solo documento."""
    relevance: str = Field(description="¿Es relevante? 'yes' o 'no'")

class GradeHallucination(BaseModel):
    """Calificación de alucinación."""
    binary_score: str = Field(description="¿La respuesta es fiel a los documentos? 'yes' o 'no'")

class GradeAnswer(BaseModel):
    """Calificación de utilidad."""
    binary_score: str = Field(description="¿Resuelve la pregunta? 'yes' o 'no'")


class AuditorNodes:
    """Nodos del grafo del Auditor Legal."""
    
    def __init__(self):
        # LLM: Amazon Nova Lite via AWS Bedrock
        self.llm = ChatBedrock(
            model_id=Config.LLM_MODEL_ID,
            provider=Config.LLM_PROVIDER,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            aws_session_token=Config.AWS_SESSION_TOKEN if Config.AWS_SESSION_TOKEN else None,
            region_name=Config.AWS_REGION,
        )
        
        # Embeddings: Amazon Titan Text v2
        self.embeddings = BedrockEmbeddings(
            model_id=Config.EMBEDDING_MODEL_ID,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            aws_session_token=Config.AWS_SESSION_TOKEN if Config.AWS_SESSION_TOKEN else None,
            region_name=Config.AWS_REGION,
            normalize=True,
        )

        # --- DEFINICIÓN DE PROMPTS Y CADENAS ---
        # 1. Calificador de documentos (uno por uno para evitar errores con structured_output)
        self.doc_grader = ChatPromptTemplate.from_template(
            """Evalúa la relevancia del siguiente documento para la pregunta legal.
            Pregunta: {question}
            Documento: {doc_text}
            ¿Es relevante? Responde solo 'yes' o 'no'."""
        ) | self.llm.with_structured_output(GradeSingle)

        # 2. Otros calificadores
        self.hallucination_grader = ChatPromptTemplate.from_template(
            "Analiza los documentos y la generación. ¿La generación es fiel a los documentos? Responde solo 'yes' o 'no'.\nDocumentos: {documents}\nGeneración: {generation}"
        ) | self.llm.with_structured_output(GradeHallucination)

        self.answer_grader = ChatPromptTemplate.from_template(
            "Pregunta: {question}\nRespuesta: {generation}.\n¿Resuelve la pregunta? Responde solo 'yes' o 'no'."
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
        """Recupera documentos relevantes de ChromaDB."""
        logger.info(f"--- RECUPERANDO PARA: {state['question']} ---")
        docs = self.retriever.invoke(state["question"])
        self._wait_for_rate_limit()
        return {"documents": docs, "steps": state["steps"] + ["retrieve"]}

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Califica la relevancia de los documentos recuperados (uno por uno)."""
        logger.info("--- CALIFICANDO DOCUMENTOS ---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        is_relevant = "no"
        
        for i, doc in enumerate(documents):
            try:
                result = self.doc_grader.invoke({
                    "question": question,
                    "doc_text": doc.page_content[:2000]  # Limitar longitud
                })
                
                if result.relevance.lower() == "yes":
                    filtered_docs.append(doc)
                    is_relevant = "yes"
                    logger.info(f"  Doc {i+1}: ✅ Relevante")
                else:
                    logger.info(f"  Doc {i+1}: ❌ No relevante")
                    
            except Exception as e:
                logger.warning(f"  Error calificando doc {i+1}: {e}. Se omite.")

        self._wait_for_rate_limit()
        logger.info(f"Documentos relevantes: {len(filtered_docs)}/{len(documents)}")
        return {"documents": filtered_docs, "is_relevant": is_relevant, "steps": state["steps"] + ["grade_documents"]}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Genera respuesta basada en los documentos relevantes."""
        logger.info("--- GENERANDO RESPUESTA ---")
        if not state["documents"]:
            return {
                "generation": "No se encontró información relevante en los documentos disponibles.",
                "steps": state["steps"] + ["generate"]
            }
        
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        prompt = f"Contexto Legal:\n{context}\n\nPregunta: {state['question']}\nRespuesta profesional:"
        generation = self.llm.invoke(prompt).content
        self._wait_for_rate_limit()
        return {"generation": generation, "steps": state["steps"] + ["generate"]}

    def transform_query(self, state: AgentState) -> Dict[str, Any]:
        """Optimiza la consulta para mejorar la recuperación."""
        logger.info("--- OPTIMIZANDO CONSULTA ---")
        prompt = f"Optimiza esta consulta legal para búsqueda en Colombia: {state['question']}"
        better_query = self.llm.invoke(prompt).content
        self._wait_for_rate_limit()
        return {"question": better_query, "retries": state["retries"] + 1, "steps": state["steps"] + ["transform_query"]}

    def grade_generation_v_documents(self, state: AgentState) -> Dict[str, Any]:
        """Valida que la respuesta no tenga alucinaciones."""
        logger.info("--- VALIDANDO ALUCINACIONES ---")
        if not state["documents"]:
            return {"hallucination": "no", "steps": state["steps"] + ["grade_hallucination"]}
        
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        score = self.hallucination_grader.invoke({"documents": context, "generation": state["generation"]})
        self._wait_for_rate_limit()
        return {"hallucination": score.binary_score, "steps": state["steps"] + ["grade_hallucination"]}

    def grade_generation_v_question(self, state: AgentState) -> Dict[str, Any]:
        """Valida que la respuesta sea útil para la pregunta."""
        logger.info("--- VALIDANDO UTILIDAD ---")
        score = self.answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        self._wait_for_rate_limit()
        return {"answer_relevant": score.binary_score, "steps": state["steps"] + ["grade_answer"]}
