# Módulo 11: Patrones Avanzados de Agentes

> **Basado en**: [Documentación Oficial de LangGraph - Agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag)  
> **Estado**: ✅ COMPLETADO - Código de Producción  
> **Prerrequisitos**: Módulos 9-10 completados

---

## 📋 Índice del Módulo

1. [11.1 - Agentic RAG (RAG como Agente)](#111-agentic-rag-rag-como-agente)
2. [11.2 - ReAct Pattern (Reasoning + Acting)](#112-react-pattern-reasoning--acting)
3. [11.3 - Multi-Agent Systems](#113-multi-agent-systems)
4. [11.4 - Reflection Pattern](#114-reflection-pattern)
5. [11.5 - Supervisor Pattern](#115-supervisor-pattern)

---

## 11.1 - Agentic RAG (RAG como Agente)

### ¿Qué es Agentic RAG?

**Agentic RAG** transforma el RAG tradicional en un agente que:
- Decide **cuándo** hacer retrieval
- **Evalúa** si los documentos son relevantes
- **Reformula** la pregunta si es necesario
- **Genera** respuesta basada en contexto

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTIC RAG                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query → Generar Query → Retrieval                      │
│                      ↓                                  │
│           ┌──────────────────┐                          │
│           │ ¿Documentos      │                          │
│           │ Relevantes?      │                          │
│           └────────┬─────────┘                          │
│                    ↓                                    │
│      ┌─────────────┴─────────────┐                      │
│      ↓                           ↓                      │
│   ✅ Sí                       ❌ No                      │
│      ↓                           ↓                      │
│  Generar                   Reformular                   │
│  Respuesta                 Query                        │
│      ↓                           ↓                      │
│      └───────────┬───────────────┘                      │
│                  ↓                                      │
│           ┌──────────────┐                              │
│           │ ¿Max Iters?  │                              │
│           └──────┬───────┘                              │
│                  ↓                                      │
│      ┌───────────┴───────────┐                          │
│      ↓                       ↓                          │
│   No → Volver a retrieval   Sí → Responder sin docs    │
└─────────────────────────────────────────────────────────┘
```

### Flujo del Agente

```python
# Nodos del grafo:
1. generate_query()      → Genera query optimizada
2. retrieve_documents()  → Retrieval del vector store
3. grade_documents()     → Evalúa relevancia (0-1)
4. transform_query()     → Reformula si es necesario
5. generate_answer()     → Genera respuesta final
```

---

## 11.2 - ReAct Pattern (Reasoning + Acting)

### ¿Qué es ReAct?

**ReAct** = **Re**asoning + **Act**ing

El agente:
1. **Piensa** (Thought): Analiza la situación
2. **Actúa** (Action): Ejecuta una herramienta
3. **Observa** (Observation): Ve el resultado
4. **Repite** hasta resolver

```
┌─────────────────────────────────────────────────────────┐
│                    REACT PATTERN                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: "¿Qué es una tutela y quién puede interponerla?"│
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Thought 1: Necesito buscar información sobre    │   │
│  │           tutela y sus requisitos               │   │
│  │ Action 1:  buscar_ley("tutela")                 │   │
│  │ Observation 1: "La tutela es un mecanismo..."   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Thought 2: Ahora necesito saber quién puede     │   │
│  │           interponerla                          │   │
│  │ Action 2:  buscar_requisitos("tutela")          │   │
│  │ Observation 2: "Cualquier persona puede..."     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Final Answer: La tutela es... y puede ser interpuesta…│
└─────────────────────────────────────────────────────────┘
```

---

## 11.3 - Multi-Agent Systems

### ¿Qué es Multi-Agent?

**Multi-Agent** usa múltiples agentes especializados que colaboran:

```
┌─────────────────────────────────────────────────────────┐
│                 MULTI-AGENT SYSTEM                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    ┌─────────────┐                      │
│                    │  Supervisor │                      │
│                    └──────┬──────┘                      │
│                           ↓                             │
│         ┌─────────────────┼─────────────────┐           │
│         ↓                 ↓                 ↓           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Agente    │  │   Agente    │  │   Agente    │     │
│  │  Búsqueda   │  │  Análisis   │  │  Redacción  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┴────────────────┘             │
│                           ↓                             │
│                    ┌─────────────┐                      │
│                    │  Respuesta  │                      │
│                    │  Final      │                      │
│                    └─────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### Casos de Uso

| Rol del Agente | Responsabilidad | Herramientas |
|----------------|-----------------|--------------|
| **Investigador** | Búsqueda de información | Vector search, web search |
| **Analista** | Análisis crítico | LLM analysis, validation |
| **Redactor** | Generar respuesta | LLM generation |
| **Crítico** | Revisar calidad | Evaluation, fact-checking |

---

## 11.4 - Reflection Pattern

### ¿Qué es Reflection?

**Reflection** permite al agente auto-evaluarse y mejorar:

```
┌─────────────────────────────────────────────────────────┐
│                  REFLECTION PATTERN                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Generate → Critique → Refine → Repeat                  │
│                                                         │
│  1. Generate: Crear respuesta inicial                   │
│  2. Critique: Evaluar y encontrar errores               │
│  3. Refine: Mejorar basado en crítica                   │
│  4. Repeat: Hasta alcanzar calidad deseada              │
└─────────────────────────────────────────────────────────┘
```

### Criterios de Reflexión

```python
# El agente crítico evalúa:
criteria = {
    "precision": "¿La información es precisa?",
    "completeness": "¿Responde completamente la pregunta?",
    "clarity": "¿Es clara y comprensible?",
    "relevance": "¿Es relevante al contexto?",
    "citation": "¿Cita las fuentes correctamente?"
}
```

---

## 11.5 - Supervisor Pattern

### ¿Qué es Supervisor?

**Supervisor** coordina múltiples agentes y decide el flujo:

```python
# Responsabilidades del Supervisor:
1. Recibir tarea del usuario
2. Descomponer en sub-tareas
3. Asignar a agentes especializados
4. Consolidar resultados
5. Entregar respuesta final
```

---

## 🎯 Ejercicio Práctico: Implementación Completa

```python
# Ver archivo: src/course_examples/modulo_11/01_patrones_avanzados.py
```

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [Agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [Multi-Agent Systems](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [ReAct Pattern](https://docs.langchain.com/oss/python/langgraph/react)

### Siguiente Módulo
➡️ **Módulo 12: Producción y Deploy**

---

*Módulo creado: 2026-03-29*  
*Código listo para producción*
