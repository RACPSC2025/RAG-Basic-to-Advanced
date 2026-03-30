"""
Compilación del grafo RAG Legal — Proyecto Fénix.

Flujo CRAG + Self-RAG:

  START
    │
    ▼
  [retrieve]  ──→  [grade_documents]
                        │
              ┌─────────┴──────────┐
           "útil"             "no_útil"
              │                    │
              ▼                    ▼
          [generate]          [no_answer]
              │                    │
              ▼                   END
    [check_hallucination]
          │           │
       "útil"    "alucinación"
          │           │
         END    [generate] ← (re-intento, max 2)
                    │
                   END
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from proyectos.Rag_Legal.nodes import (
    check_hallucination,
    generate,
    grade_documents,
    no_answer,
    retrieve,
)
from proyectos.Rag_Legal.state import RagState

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Edges condicionales
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_grade(
    state: RagState,
) -> Literal["generate", "no_answer"]:
    """Enruta después de la evaluación de relevancia."""
    if state.grade == "útil" and state.documents:
        logger.info("[ROUTER] Documentos relevantes → generate.")
        return "generate"
    logger.info("[ROUTER] Sin docs relevantes → no_answer.")
    return "no_answer"


def route_after_hallucination(
    state: RagState,
) -> Literal["__end__", "generate"]:
    """
    Enruta después del verificador de alucinaciones.
    Si se detectó alucinación y quedan intentos, re-genera.
    """
    if state.grade == "útil":
        logger.info("[ROUTER] Respuesta verificada → END.")
        return END

    if state.attempts < MAX_ATTEMPTS:
        logger.warning(
            f"[ROUTER] Alucinación detectada, reintento {state.attempts}/{MAX_ATTEMPTS} → generate."
        )
        return "generate"

    logger.error("[ROUTER] Máximo de intentos alcanzado. Entregando respuesta con advertencia.")
    return END


# ═══════════════════════════════════════════════════════════════════════════════
# Construcción del grafo
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Construye y compila el StateGraph del RAG Legal."""
    graph = StateGraph(RagState)

    # Nodos
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("no_answer", no_answer)

    # Flujo principal
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # Routing tras evaluación
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "no_answer": "no_answer"},
    )

    # Después de generar → verificar alucinaciones
    graph.add_edge("generate", "check_hallucination")

    # Routing tras verificación
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {END: END, "generate": "generate"},
    )

    # No answer siempre termina
    graph.add_edge("no_answer", END)

    return graph.compile()


# Singleton compilado para reutilizar entre llamadas en Streamlit
_compiled_graph = None


def get_graph():
    """Retorna el grafo compilado (singleton thread-safe para Streamlit)."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Compilando grafo RAG Legal...")
        _compiled_graph = build_graph()
        logger.info("Grafo compilado correctamente.")
    return _compiled_graph


def query(question: str) -> dict:
    """
    Punto de entrada principal del RAG Legal.

    Args:
        question: Consulta legal en lenguaje natural.

    Returns:
        dict con 'answer', 'source_docs', 'attempts' y 'grade'.
    """
    app = get_graph()
    initial_state = RagState(question=question)

    final_state: RagState = app.invoke(initial_state)

    return {
        "answer": final_state.generation,
        "source_docs": final_state.source_docs,
        "attempts": final_state.attempts,
        "grade": final_state.grade,
        "hallucination_score": final_state.hallucination_score,
    }
