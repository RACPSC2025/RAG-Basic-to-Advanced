from langgraph.graph import END, StateGraph, START
from .state import AgentState
from .nodes import AuditorNodes
from ..config import Config

def create_auditor_graph():
    nodes = AuditorNodes()
    workflow = StateGraph(AgentState)

    # 1. Definir Nodos
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)
    workflow.add_node("grade_hallucination", nodes.grade_generation_v_documents)
    workflow.add_node("grade_answer", nodes.grade_generation_v_question)

    # 2. Definir Aristas (Edges) y Flujo
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Lógica Condicional: ¿Hay documentos relevantes?
    def decide_to_generate(state):
        if state["is_relevant"] == "yes":
            return "generate"
        if state["retries"] >= Config.MAX_RETRIES:
            return "generate"  # Finalizar con lo que tenga tras reintentos
        return "transform_query"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query"
        }
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "grade_hallucination")

    # Lógica Condicional: ¿Alucinación?
    def decide_hallucination(state):
        if state["hallucination"] == "yes":
            return "generate"  # Re-generar si hay alucinación
        return "grade_answer"

    workflow.add_conditional_edges(
        "grade_hallucination",
        decide_hallucination,
        {
            "generate": "generate",
            "grade_answer": "grade_answer"
        }
    )

    # Lógica Condicional: ¿Respuesta útil?
    def decide_answer_quality(state):
        if state["answer_relevant"] == "yes":
            return END
        return "transform_query" # Si no es útil, intentar buscar mejor

    workflow.add_conditional_edges(
        "grade_answer",
        decide_answer_quality,
        {
            "END": END,
            "transform_query": "transform_query"
        }
    )

    return workflow.compile()
