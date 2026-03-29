"""
01_primer_grafo.py
Tu primer grafo con LangGraph

Objetivo: Entender State, Nodes y Edges
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

load_dotenv()


# ============================================================
# 1. DEFINIR EL ESTADO
# ============================================================

class State(TypedDict):
    """Define la estructura del estado del grafo"""
    messages: List[str]
    contador: int


# ============================================================
# 2. CREAR LOS NODOS
# ============================================================

def node_saludo(state: State) -> State:
    """Primer nodo: Saludar"""
    print("👋 Nodo: Saludo")
    
    messages = state["messages"] + ["¡Hola! Bienvenido al grafo."]
    
    return {
        "messages": messages,
        "contador": state["contador"] + 1
    }


def node_despedida(state: State) -> State:
    """Segundo nodo: Despedir"""
    print("👋 Nodo: Despedida")
    
    messages = state["messages"] + ["¡Adiós! Gracias por usar el grafo."]
    
    return {
        "messages": messages,
        "contador": state["contador"] + 1
    }


def node_procesar(state: State) -> State:
    """Tercer nodo: Procesar"""
    print("⚙️ Nodo: Procesar")
    
    messages = state["messages"] + ["Procesando información..."]
    
    return {
        "messages": messages,
        "contador": state["contador"] + 1
    }


# ============================================================
# 3. CONSTRUIR EL GRAFO
# ============================================================

def construir_grafo_simple():
    """Construir un grafo simple secuencial"""
    
    print("=" * 60)
    print("GRAFO SIMPLE SECUENCIAL")
    print("=" * 60)
    
    builder = StateGraph(State)
    
    # Agregar nodos
    builder.add_node("saludo", node_saludo)
    builder.add_node("procesar", node_procesar)
    builder.add_node("despedida", node_despedida)
    
    # Agregar edges (conexiones)
    builder.add_edge(START, "saludo")
    builder.add_edge("saludo", "procesar")
    builder.add_edge("procesar", "despedida")
    builder.add_edge("despedida", END)
    
    # Compilar el grafo
    graph = builder.compile()
    
    print("\n✅ Grafo construido exitosamente")
    print(f"📊 Estructura: START → saludo → procesar → despedida → END")
    
    return graph


def ejecutar_grafo_simple():
    """Ejecutar el grafo simple"""
    
    graph = construir_grafo_simple()
    
    initial_state = {
        "messages": [],
        "contador": 0
    }
    
    print("\n--- Ejecutando Grafo ---")
    print(f"Estado inicial: {initial_state}")
    
    final_state = graph.invoke(initial_state)
    
    print(f"\nEstado final: {final_state}")
    print(f"📝 Messages: {final_state['messages']}")
    print(f"🔢 Contador: {final_state['contador']}")


# ============================================================
# 4. GRAFO CON DECISIÓN (CONDITIONAL EDGES)
# ============================================================

class StateConDecision(TypedDict):
    """Estado con campo de decisión"""
    mensaje: str
    respuesta: str
    paso: int


def node_verificar(state: StateConDecision) -> StateConDecision:
    """Nodo que verifica el mensaje"""
    print("🔍 Nodo: Verificar")
    
    mensaje = state["mensaje"].lower()
    
    if "hola" in mensaje or "buenos" in mensaje:
        decision = "saludar"
    elif "adios" in mensaje or "chao" in mensaje:
        decision = "despedir"
    else:
        decision = "procesar"
    
    print(f"   Decisión: {decision}")
    
    return {
        **state,
        "respuesta": decision,
        "paso": state["paso"] + 1
    }


def node_saludar_condicional(state: StateConDecision) -> StateConDecision:
    """Nodo para saludar"""
    print("👋 Nodo: Saludar")
    
    return {
        **state,
        "respuesta": "¡Hola! ¿Cómo estás?",
        "paso": state["paso"] + 1
    }


def node_despedir_condicional(state: StateConDecision) -> StateConDecision:
    """Nodo para despedir"""
    print("👋 Nodo: Despedir")
    
    return {
        **state,
        "respuesta": "¡Hasta luego!",
        "paso": state["paso"] + 1
    }


def node_procesar_condicional(state: StateConDecision) -> StateConDecision:
    """Nodo para procesar mensaje normal"""
    print("⚙️ Nodo: Procesar")
    
    return {
        **state,
        "respuesta": f"Procesando: {state['mensaje']}",
        "paso": state["paso"] + 1
    }


def construir_grafo_con_decision():
    """Construir grafo con decisión condicional"""
    
    print("\n" + "=" * 60)
    print("GRAFO CON DECISIÓN CONDICIONAL")
    print("=" * 60)
    
    builder = StateGraph(StateConDecision)
    
    # Agregar nodos
    builder.add_node("verificar", node_verificar)
    builder.add_node("saludar", node_saludar_condicional)
    builder.add_node("procesar", node_procesar_condicional)
    builder.add_node("despedir", node_despedir_condicional)
    
    # Edge inicial
    builder.add_edge(START, "verificar")
    
    # Conditional edge (decisión)
    def router(state: StateConDecision) -> str:
        """Función que decide el siguiente nodo"""
        return state["respuesta"]
    
    builder.add_conditional_edges(
        "verificar",
        router,
        {
            "saludar": "saludar",
            "procesar": "procesar",
            "despedir": "despedir"
        }
    )
    
    # Todos terminan en END
    builder.add_edge("saludar", END)
    builder.add_edge("procesar", END)
    builder.add_edge("despedir", END)
    
    graph = builder.compile()
    
    print("\n✅ Grafo con decisión construido")
    print(f"📊 Estructura: START → verificar → [saludar|procesar|despedir] → END")
    
    return graph


def ejecutar_grafo_con_decision(mensaje: str):
    """Ejecutar grafo con decisión"""
    
    graph = construir_grafo_con_decision()
    
    initial_state = {
        "mensaje": mensaje,
        "respuesta": "",
        "paso": 0
    }
    
    print(f"\n--- Ejecutando con mensaje: '{mensaje}' ---")
    
    final_state = graph.invoke(initial_state)
    
    print(f"\nResultado: {final_state['respuesta']}")
    print(f"Pasos: {final_state['paso']}")


# ============================================================
# 5. GRAFO CON BUCLE (LOOP)
# ============================================================

class StateConBucle(TypedDict):
    """Estado con bucle"""
    numero: int
    historial: List[str]


def node_incrementar(state: StateConBucle) -> StateConBucle:
    """Nodo que incrementa el número"""
    print(f"🔢 Nodo: Incrementar (actual: {state['numero']})")
    
    nuevo_numero = state["numero"] + 1
    
    return {
        "numero": nuevo_numero,
        "historial": state["historial"] + [f"Incrementado a {nuevo_numero}"]
    }


def node_verificar_limite(state: StateConBucle) -> StateConBucle:
    """Nodo que verifica si llegó al límite"""
    print(f"🔍 Nodo: Verificar límite ({state['numero']})")
    
    return state


def construir_grafo_con_bucle():
    """Construir grafo con bucle"""
    
    print("\n" + "=" * 60)
    print("GRAFO CON BUCLE (LOOP)")
    print("=" * 60)
    
    builder = StateGraph(StateConBucle)
    
    # Nodos
    builder.add_node("incrementar", node_incrementar)
    builder.add_node("verificar", node_verificar_limite)
    
    # Edges
    builder.add_edge(START, "incrementar")
    builder.add_edge("incrementar", "verificar")
    
    # Conditional edge con bucle
    def debe_continuar(state: StateConBucle) -> str:
        if state["numero"] < 5:
            return "continuar"
        else:
            return "terminar"
    
    builder.add_conditional_edges(
        "verificar",
        debe_continuar,
        {
            "continuar": "incrementar",
            "terminar": END
        }
    )
    
    graph = builder.compile()
    
    print("\n✅ Grafo con bucle construido")
    print(f"📊 Estructura: START → incrementar → verificar → [↫incrementar|END]")
    
    return graph


def ejecutar_grafo_con_bucle():
    """Ejecutar grafo con bucle"""
    
    graph = construir_grafo_con_bucle()
    
    initial_state = {
        "numero": 0,
        "historial": []
    }
    
    print("\n--- Ejecutando Bucle ---")
    
    final_state = graph.invoke(initial_state)
    
    print(f"\nResultado final: {final_state['numero']}")
    print(f"Historial: {final_state['historial']}")


if __name__ == "__main__":
    # Grafo simple
    ejecutar_grafo_simple()
    
    # Grafo con decisión
    print("\n" + "=" * 80)
    ejecutar_grafo_con_decision("Hola")
    print("\n" + "=" * 80)
    ejecutar_grafo_con_decision("Adiós")
    print("\n" + "=" * 80)
    ejecutar_grafo_con_decision("Necesito ayuda")
    
    # Grafo con bucle
    print("\n" + "=" * 80)
    ejecutar_grafo_con_bucle()
