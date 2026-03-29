# Módulo 4.1: Introducción a LangGraph

## Objetivos
- Comprender qué es LangGraph y por qué usarlo
- Entender los conceptos de State, Nodes y Edges
- Crear tu primer grafo con StateGraph
- Ejecutar y visualizar el grafo

---

## 4.1.1 ¿Qué es LangGraph?

**LangGraph** es una librería para construir agentes como **grafos de estado**.

### LangChain vs LangGraph

| Aspecto | LangChain | LangGraph |
|---------|-----------|-----------|
| **Flujo** | Secuencial (chains) | Cíclico (grafos) |
| **Estado** | Implícito | Explícito |
| **Decisiones** | Fijas | Dinámicas |
| **Uso ideal** | Flujos simples | Agentes complejos |

```
┌─────────────────────────────────────────────────────────┐
│              LANGCHAIN (Sequential)                     │
├─────────────────────────────────────────────────────────┤
│  Prompt → LLM → Parser → Output                         │
│                                                         │
│  ✅ Simple, directo                                     │
│  ❌ Sin bucles, sin decisiones dinámicas                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              LANGGRAPH (Graph-based)                    │
├─────────────────────────────────────────────────────────┤
│           ┌─────────────┐                               │
│           │   Start     │                               │
│           └──────┬──────┘                               │
│                  ▼                                      │
│           ┌─────────────┐                               │
│           │  LLM Node   │◄──────┐                       │
│           └──────┬──────┘       │                       │
│                  ▼              │                       │
│           ┌─────────────┐       │                       │
│           │  Decision   │───────┘ (loop)                │
│           └──────┬──────┘                               │
│                  ▼                                      │
│           ┌─────────────┐                               │
│           │    End      │                               │
│           └─────────────┘                               │
│                                                         │
│  ✅ Bucles, decisiones, estado compartido               │
│  ❌ Más complejo                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 4.1.2 Conceptos Fundamentales

### 1. State (Estado)

El **estado** es un diccionario que contiene toda la información del grafo:

```python
from typing import TypedDict, List

class State(TypedDict):
    messages: List[str]
    contador: int
```

### 2. Nodes (Nodos)

Los **nodes** son funciones que procesan el estado:

```python
def mi_node(state: State) -> State:
    # Procesa el estado
    return {"messages": ["Hola"]}
```

### 3. Edges (Aristas)

Los **edges** conectan nodes y definen el flujo:

```python
graph.add_edge("node1", "node2")  # node1 → node2
```

### 4. Conditional Edges

Edges que deciden el siguiente node basado en el estado:

```python
graph.add_conditional_edges(
    "node1",
    lambda state: "si" if condition else "no",
    {"si": "node2", "no": "node3"}
)
```

---

## 4.1.3 Código de Ejemplo

Archivo: `src/course_examples/modulo_04/01_primer_grafo.py`

```python
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
    
    # Crear el builder del grafo
    builder = StateGraph(State)
    
    # Agregar nodos
    builder.add_node("saludo", node_saludo)
    builder.add_node("procesar", node_procesar)
    builder.add_node("despedida", node_despedida)
    
    # Agregar edges (conexiones)
    builder.add_edge(START, "saludo")        # Start → saludo
    builder.add_edge("saludo", "procesar")   # saludo → procesar
    builder.add_edge("procesar", "despedida") # procesar → despedida
    builder.add_edge("despedida", END)       # despedida → End
    
    # Compilar el grafo
    graph = builder.compile()
    
    # Visualizar (si tienes graphviz instalado)
    # graph.draw_mermaid_png()
    
    print("\n✅ Grafo construido exitosamente")
    print(f"📊 Estructura: START → saludo → procesar → despedida → END")
    
    return graph


def ejecutar_grafo_simple():
    """Ejecutar el grafo simple"""
    
    graph = construir_grafo_simple()
    
    # Estado inicial
    initial_state = {
        "messages": [],
        "contador": 0
    }
    
    print("\n--- Ejecutando Grafo ---")
    print(f"Estado inicial: {initial_state}")
    
    # Ejecutar
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
        return state["respuesta"]  # Retorna: "saludar", "procesar", o "despedir"
    
    builder.add_conditional_edges(
        "verificar",           # Desde este nodo
        router,                # Función que decide
        {                      # Mapeo de decisiones a nodos
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
            "continuar": "incrementar",  # ← Bucle!
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


# ============================================================
# 6. VISUALIZACIÓN DEL GRAFO
# ============================================================

def visualizar_grafo():
    """Mostrar cómo visualizar el grafo"""
    
    print("\n" + "=" * 60)
    print("VISUALIZACIÓN DEL GRAFO")
    print("=" * 60)
    
    graph = construir_grafo_con_decision()
    
    # LangGraph permite visualizar el grafo
    # Opción 1: Mermaid
    print("\nDiagrama Mermaid:")
    print(graph.get_graph().draw_mermaid())
    
    # Opción 2: PNG (requiere graphviz)
    # graph.get_graph().draw_png("grafo.png")
    
    # Opción 3: ASCII
    print("\nEstructura ASCII:")
    for nodo in graph.get_graph().nodes:
        print(f"  - {nodo}")


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
    
    # Visualización
    visualizar_grafo()

```

---

## 4.1.4 Elementos Clave de LangGraph

### StateGraph

```python
from langgraph.graph import StateGraph

builder = StateGraph(State)  # Define el tipo de estado
```

### Nodes

```python
builder.add_node("nombre", funcion)  # Agrega un nodo
```

### Edges

```python
builder.add_edge("origen", "destino")  # Conexión simple
```

### Conditional Edges

```python
builder.add_conditional_edges(
    "origen",
    funcion_router,  # Retorna string con el siguiente nodo
    {"opcion1": "nodo1", "opcion2": "nodo2"}
)
```

### START y END

```python
from langgraph.graph import START, END

builder.add_edge(START, "primer_nodo")
builder.add_edge("ultimo_nodo", END)
```

---

## 4.1.5 Ejercicios Prácticos

### Ejercicio 1: Calculadora

Crea un grafo que:
- Tenga un estado con `numero` y `operacion`
- Incremente o decremente según la operación
- Termine cuando llegue a un objetivo

### Ejercicio 2: Clasificador

Crea un grafo que:
- Clasifique mensajes como "positivo", "negativo", "neutral"
- Route a diferentes nodos según la clasificación
- Retorne una respuesta apropiada

### Ejercicio 3: Aprobación

Crea un grafo que:
- Procesa una solicitud
- Pide aprobación humana
- Si aprueba: ejecuta; si rechaza: cancela

---

## 4.1.6 Recursos Adicionales

### Documentación Oficial
- [LangGraph Introduction](https://docs.langchain.com/oss/python/langgraph)
- [StateGraph](https://docs.langchain.com/oss/python/langgraph/concepts/stategraph)
- [Nodes & Edges](https://docs.langchain.com/oss/python/langgraph/concepts/low_level)

### Siguiente Lección
➡️ **4.2 Conditional Routing**

---

*Lección creada: 2026-03-29*
