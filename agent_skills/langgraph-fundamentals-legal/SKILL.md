---
name: langgraph-fundamentals-legal
description: |
  Skill fundamental de LangGraph para construir agentes legales.
  Cubre StateGraph, nodes, edges, conditional routing y state management.
  Adaptado para flujos de trabajo legales con énfasis en trazabilidad y control.
---

# LangGraph Fundamentals Legal Skill

Este skill proporciona conocimiento para construir agentes legales usando LangGraph.

## Cuándo Usar Este Skill

Usa este skill cuando necesites:

- Construir flujos de trabajo legales multi-paso
- Implementar routing condicional basado en tipo de documento
- Mantener estado durante conversaciones legales largas
- Crear agentes que toman decisiones basadas en reglas legales
- Implementar bucles de refinamiento de documentos

## Conceptos Clave

### StateGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
import operator

class LegalState(TypedDict):
    """Estado para flujo legal."""
    documento: str
    tipo_legal: str  # tutela, demanda, sentencia, contrato
    analisis: str
    riesgos: List[str]
    recomendaciones: List[str]
    mensajes: Annotated[List, operator.add]  # Acumulativo
```

### Nodes

```python
def node_clasificar(state: LegalState) -> dict:
    """Clasifica el tipo de documento legal."""
    documento = state['documento']
    
    # Lógica de clasificación
    if "tutela" in documento.lower():
        tipo = "tutela"
    elif "demanda" in documento.lower():
        tipo = "demanda"
    else:
        tipo = "otro"
    
    return {"tipo_legal": tipo}

def node_analizar_riesgos(state: LegalState) -> dict:
    """Analiza riesgos legales del documento."""
    # Análisis con LLM
    riesgos = identificar_riesgos(state['documento'])
    return {"riesgos": riesgos}
```

### Edges Condicionales

```python
def router_tipo_documento(state: LegalState) -> str:
    """Decide qué ruta tomar según el tipo de documento."""
    tipo = state['tipo_legal']
    
    if tipo == "tutela":
        return "ruta_tutela"
    elif tipo == "demanda":
        return "ruta_demanda"
    else:
        return "ruta_general"

# Construir grafo
builder = StateGraph(LegalState)

builder.add_node("clasificar", node_clasificar)
builder.add_node("analizar_tutela", node_analizar_tutela)
builder.add_node("analizar_demanda", node_analizar_demanda)
builder.add_node("analisis_general", node_analisis_general)

builder.add_edge(START, "clasificar")
builder.add_conditional_edges(
    "clasificar",
    router_tipo_documento,
    {
        "ruta_tutela": "analizar_tutela",
        "ruta_demanda": "analizar_demanda",
        "ruta_general": "analisis_general"
    }
)

builder.add_edge("analizar_tutela", END)
builder.add_edge("analizar_demanda", END)
builder.add_edge("analisis_general", END)

graph = builder.compile()
```

## Patrones Comunes

### 1. Clasificador y Enrutador

```
Input → Clasificador → Router → Nodo Especializado → Output
```

### 2. Análisis Multi-Fase

```
Input → Fase 1 (Extraer) → Fase 2 (Analizar) → Fase 3 (Validar) → Output
```

### 3. Bucle de Refinamiento

```
Input → Generar → Evaluar → ¿Calidad? → No → Refinar → (bucle)
                              ↓
                             Sí → Output
```

## State Management

### Reducers Personalizados

```python
from typing import TypedDict, Annotated
import operator

def agregar_riesgo(existing: list, new: str) -> list:
    """Agrega un riesgo solo si no existe."""
    if new not in existing:
        return existing + [new]
    return existing

class LegalState(TypedDict):
    riesgos: Annotated[list, agregar_riesgo]
    cambios: Annotated[list, operator.add]  # Acumula todos
```

## Ejemplo: Flujo de Análisis de Contratos

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class ContratoState(TypedDict):
    texto: str
    clausulas: List[str]
    riesgos: List[str]
    sugerencias: List[str]
    aprobado: bool

def extraer_clausulas(state: ContratoState) -> dict:
    """Extrae cláusulas del contrato."""
    # Implementación
    return {"clausulas": [...]}

def identificar_riesgos(state: ContratoState) -> dict:
    """Identifica riesgos en cada cláusula."""
    # Implementación
    return {"riesgos": [...]}

def generar_sugerencias(state: ContratoState) -> dict:
    """Genera sugerencias de mejora."""
    # Implementación
    return {"sugerencias": [...]}

def decision_aprobacion(state: ContratoState) -> str:
    """Decide si el contrato está aprobado."""
    if len(state['riesgos']) == 0:
        return "aprobado"
    else:
        return "requiere_revision"

# Construir grafo
builder = StateGraph(ContratoState)

builder.add_node("extraer", extraer_clausulas)
builder.add_node("analizar_riesgos", identificar_riesgos)
builder.add_node("generar_sugerencias", generar_sugerencias)
builder.add_node("aprobado", lambda s: {"aprobado": True})
builder.add_node("revision", lambda s: {"aprobado": False})

builder.add_edge(START, "extraer")
builder.add_edge("extraer", "analizar_riesgos")
builder.add_edge("analizar_riesgos", "generar_sugerencias")

builder.add_conditional_edges(
    "generar_sugerencias",
    decision_aprobacion,
    {
        "aprobado": "aprobado",
        "requiere_revision": "revision"
    }
)

builder.add_edge("aprobado", END)
builder.add_edge("revision", END)

graph = builder.compile()
```

## Métricas de Calidad

| Métrica | Objetivo | Cómo Medir |
|---------|----------|------------|
| Precisión de Routing | >95% | Correct routes / Total routes |
| Tiempo de Ejecución | <10s | Average execution time |
| State Consistency | 100% | No state corruption incidents |

## Referencias Cruzadas

- `langchain-rag-legal` - Para retrieval en flujos de LangGraph
- `langgraph-human-in-the-loop` - Para aprobación humana en nodos críticos
- `deep-agents-orchestration` - Para orquestación avanzada
