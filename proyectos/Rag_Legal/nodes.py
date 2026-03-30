"""
Nodos del grafo RAG Legal — Proyecto Fénix.

Patrón: CRAG (Corrective RAG) + Self-RAG
  Node 1 → retrieve:           Busca en ChromaDB los chunks más similares.
  Node 2 → grade_documents:    Juez LLM evalúa relevancia de cada chunk.
  Node 3 → generate:           Genera respuesta citando SOLO evidencia aprobada.
  Node 4 → check_hallucination: Verifica que cada afirmación esté en los docs.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from proyectos.Rag_Legal.config import get_llm, settings
from proyectos.Rag_Legal.ingestor import get_vector_store
from proyectos.Rag_Legal.state import RagState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Schemas de salida estructurada para el LLM
# ═══════════════════════════════════════════════════════════════════════════════

class GradeOutput(BaseModel):
    """Resultado del nodo evaluador de relevancia."""
    score: str = Field(
        description="'si' si el documento responde la pregunta, 'no' si no es relevante."
    )
    razon: str = Field(
        description="Explicación breve del veredicto (1-2 oraciones)."
    )


class HallucinationOutput(BaseModel):
    """Resultado del verificador de alucinaciones."""
    score: str = Field(
        description="'limpio' si toda afirmación está respaldada por los documentos, "
                    "'alucinacion' si hay información inventada."
    )
    razon: str = Field(
        description="Explicación de qué parte podría estar fabricada (si aplica)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NODO 1: Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(state: RagState) -> dict:
    """
    Busca en ChromaDB los TOP_K documentos semánticamente más cercanos
    a la pregunta del usuario.
    """
    logger.info(f"[RETRIEVE] Pregunta: {state.question}")
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K},
    )
    docs = retriever.invoke(state.question)
    logger.info(f"[RETRIEVE] {len(docs)} documentos recuperados.")
    return {"documents": docs}


# ═══════════════════════════════════════════════════════════════════════════════
# NODO 2: Grader de relevancia (CRAG)
# ═══════════════════════════════════════════════════════════════════════════════

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Eres un juez legal experto colombiano. Tu único trabajo es decidir si un fragmento \
de un documento normativo es RELEVANTE para responder una pregunta concreta.

REGLAS:
- Responde SOLO 'si' o 'no' en el campo score.
- Si el fragmento menciona conceptos parcialmente relacionados pero no responde la \
pregunta de forma directa, di 'no'.
- Si hay duda razonable de que el fragmento aporta información útil, di 'si'.
- Nunca inventes información. Solo evalúas lo que está en el fragmento."""
    ),
    (
        "human",
        "Pregunta: {question}\n\nFragmento del documento:\n{document}"
    ),
])


def grade_documents(state: RagState) -> dict:
    """
    Pasa cada documento por el Juez LLM.
    Solo retiene los que reciben score 'si'.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeOutput)
    chain = _GRADE_PROMPT | structured_llm

    relevant_docs: List[Document] = []
    for doc in state.documents:
        try:
            result: GradeOutput = chain.invoke({
                "question": state.question,
                "document": doc.page_content,
            })
            if result.score.strip().lower() == "si":
                relevant_docs.append(doc)
                logger.info(f"[GRADE] ✅ Relevante — {result.razon[:80]}")
            else:
                logger.info(f"[GRADE] ❌ Ignorado — {result.razon[:80]}")
        except Exception as e:
            logger.warning(f"[GRADE] Error evaluando chunk: {e}. Se omite.")

    grade_result = "útil" if relevant_docs else "no_útil"
    logger.info(f"[GRADE] Veredicto final: {grade_result} ({len(relevant_docs)}/{len(state.documents)} relevantes).")

    # Reset documents → solo los aprobados
    return {
        "documents": relevant_docs,
        "grade": grade_result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODO 3: Generación de respuesta (Self-RAG)
# ═══════════════════════════════════════════════════════════════════════════════

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Eres Fénix, un asistente legal especializado en normativa colombiana.

REGLAS ESTRICTAS — incumplirlas está PROHIBIDO:
1. Responde ÚNICAMENTE con información contenida en los DOCUMENTOS FUENTE proporcionados.
2. Si la respuesta no está en los documentos, di exactamente: \
"No dispongo de información suficiente en los documentos disponibles para responder esta pregunta."
3. Cita siempre el artículo, capítulo o sección de donde extraes la información. \
Formato: (Fuente: [nombre del documento], [Artículo/Cápitulo X]).
4. No añadas opiniones, interpretaciones o conocimiento externo.
5. Usa un lenguaje claro, formal y preciso."""
    ),
    (
        "human",
        """DOCUMENTOS FUENTE:
{context}

PREGUNTA:
{question}

RESPUESTA:"""
    ),
])


def generate(state: RagState) -> dict:
    """
    Genera la respuesta final basada exclusivamente en los documentos aprobados
    por el nodo grader.
    """
    if not state.documents:
        return {
            "generation": "No dispongo de información suficiente en los documentos disponibles para responder esta pregunta.",
            "source_docs": [],
            "attempts": state.attempts + 1,
        }

    # Construir contexto con metadata de fuente
    context_parts = []
    source_refs = []
    for doc in state.documents:
        source = doc.metadata.get("source", "documento")
        context_parts.append(f"[Fuente: {source}]\n{doc.page_content}")
        source_refs.append(f"{source}: {doc.page_content[:120]}...")

    context = "\n\n---\n\n".join(context_parts)

    llm = get_llm()
    chain = _GENERATE_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({"context": context, "question": state.question})
        logger.info(f"[GENERATE] Respuesta generada ({len(answer)} chars).")
    except Exception as e:
        logger.error(f"[GENERATE] Error del LLM: {e}")
        answer = "Error al generar la respuesta. Por favor intente nuevamente."

    return {
        "generation": answer,
        "source_docs": source_refs,
        "attempts": state.attempts + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODO 4: Verificador de alucinaciones
# ═══════════════════════════════════════════════════════════════════════════════

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Eres un verificador de precisión factual para documentos legales colombianos.

Tu tarea: comparar la RESPUESTA GENERADA contra los DOCUMENTOS FUENTE y detectar \
si la respuesta contiene afirmaciones que NO están respaldadas por los documentos.

REGLAS:
- score = 'limpio': TODA afirmación de la respuesta tiene respaldo explícito en los documentos.
- score = 'alucinacion': al menos UNA afirmación no está en los documentos o está distorsionada.
- Sé estricto. En dominio legal, una invención puede tener consecuencias graves."""
    ),
    (
        "human",
        """DOCUMENTOS FUENTE:
{documents}

RESPUESTA GENERADA:
{generation}"""
    ),
])


def check_hallucination(state: RagState) -> dict:
    """
    Verifica que la respuesta generada esté completamente respaldada
    por los documentos fuente. Si detecta alucinación, marca para re-intento.
    """
    if not state.documents:
        # Sin documentos no hay nada que verificar
        return {"grade": "no_útil", "hallucination_score": 0.0}

    # Concatenar solo los primeros 2000 chars de contexto para no sobrepasar límites
    context = "\n---\n".join(
        doc.page_content[:500] for doc in state.documents
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(HallucinationOutput)
    chain = _HALLUCINATION_PROMPT | structured_llm

    try:
        result: HallucinationOutput = chain.invoke({
            "documents": context,
            "generation": state.generation,
        })

        if result.score.strip().lower() == "limpio":
            logger.info(f"[HALLUCINATION] ✅ Respuesta verificada — {result.razon[:80]}")
            return {"grade": "útil", "hallucination_score": 0.0}
        else:
            logger.warning(f"[HALLUCINATION] ⚠️ Posible alucinación — {result.razon[:80]}")
            return {"grade": "alucinación", "hallucination_score": 0.9}

    except Exception as e:
        logger.error(f"[HALLUCINATION] Error en verificación: {e}. Se aprueba por defecto.")
        return {"grade": "útil", "hallucination_score": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# NODO 5: Respuesta de fallback (sin documentos relevantes)
# ═══════════════════════════════════════════════════════════════════════════════

def no_answer(state: RagState) -> dict:
    """Emite mensaje de fallback cuando no se encontraron documentos relevantes."""
    return {
        "generation": (
            "⚠️ No se encontró información relevante en los documentos indexados "
            "para responder su consulta legal. Le recomendamos reformular la pregunta "
            "o consultar directamente la normativa colombiana aplicable."
        ),
        "source_docs": [],
    }
