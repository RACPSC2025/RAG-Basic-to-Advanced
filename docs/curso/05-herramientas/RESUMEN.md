# Módulo 5: Herramientas (Tools) - Resumen Ejecutivo

> **Estado**: ✅ Completo  
> **Última Actualización**: 2026-03-29  
> **Documentación Base**: [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)

---

## 📋 Contenido del Módulo

| Lección | Archivo Documentación | Archivo Código |
|---------|----------------------|----------------|
| 5.1 Creación Básica | [README.md](README.md#51-creación-básica-de-herramientas) | `01_creacion_basica.py` |
| 5.2 Schema Personalizado | [README.md](README.md#52-herramientas-con-schema-personalizado) | `02_schema_personalizado.py` |
| 5.3 ToolNode LangGraph | [README.md](README.md#53-toolnode-en-langgraph) | `03_toolnode_langgraph.py` |

---

## 🎯 Objetivos de Aprendizaje

Al completar este módulo, podrás:

1. ✅ Crear herramientas con el decorador `@tool`
2. ✅ Definir schemas complejos con Pydantic
3. ✅ Usar ToolNode en LangGraph
4. ✅ Acceder y modificar estado desde herramientas
5. ✅ Implementar streaming en herramientas
6. ✅ Comparar LangChain vs LangGraph vs LlamaIndex

---

## 🔧 Conceptos Clave

### 1. Decorador @tool

```python
from langchain.tools import tool

@tool
def mi_herramienta(param: str) -> str:
    """Descripción clara para el LLM."""
    return f"Resultado: {param}"
```

**Automáticamente extrae**:
- Nombre de la función
- Descripción del docstring
- Schema de argumentos (type hints)

### 2. Schema con Pydantic

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class MiSchema(BaseModel):
    campo1: str = Field(description="Descripción del campo")
    campo2: int = Field(default=10, description="Valor por defecto")

@tool(args_schema=MiSchema)
def mi_tool(campo1: str, campo2: int = 10) -> str:
    """Herramienta con schema complejo."""
    return "resultado"
```

### 3. ToolNode en LangGraph

```python
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState

# Crear ToolNode
tool_node = ToolNode([herramienta1, herramienta2])

# En un grafo
builder = StateGraph(MessagesState)
builder.add_node("herramientas", tool_node)
```

### 4. Estado y Contexto

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def leer_estado(runtime: ToolRuntime) -> str:
    """Lee información del estado."""
    return runtime.state.get("campo", "default")

@tool
def escribir_estado(valor: str) -> Command:
    """Escribe en el estado."""
    return Command(update={"campo": valor})
```

---

## 📊 Comparativa de Frameworks

| Característica | LangChain | LangGraph | LlamaIndex |
|----------------|-----------|-----------|------------|
| **Creación** | `@tool` | `@tool` + `ToolNode` | `FunctionTool.from_defaults()` |
| **Schema** | Pydantic | Pydantic | Pydantic + Annotated |
| **Estado** | `ToolRuntime.state` | State injection | Limitado |
| **Contexto** | `ToolRuntime.context` | Context schema | ❌ No soportado |
| **Store** | `ToolRuntime.store` | BaseStore | Return direct |
| **Streaming** | `stream_writer` | Vía ToolNode | ❌ No soportado |
| **Error Handling** | ✅ Avanzado | ✅ ToolNode maneja | ⚠️ Básico |
| **Use Case** | Agentes simples | Agentes complejos | RAG + queries |

---

## 🎯 Cuándo Usar Cada Framework

### LangChain (create_agent)

```
✅ Úsalo cuando:
- Necesitas un agente simple con tools
- No requieres control total del flujo
- Quieres implementación rápida
```

### LangGraph (ToolNode + StateGraph)

```
✅ Úsalo cuando:
- Necesitas control del flujo
- Múltiples herramientas en paralelo
- Estado personalizado
- Human-in-the-loop
- Subgraphs y composición
```

### LlamaIndex (FunctionTool)

```
✅ Úsalo cuando:
- Tu caso de uso es principalmente RAG
- Queries sobre documentos
- Ya usas LlamaIndex para embeddings
- Quieres return_direct
```

---

## 💡 Mejores Prácticas

### ✅ DO (Haz esto)

```python
# 1. Type hints siempre
@tool
def mi_funcion(param: str) -> str:  # ✅

# 2. Docstring descriptivo
@tool
def buscar(nombre: str) -> str:
    """
    Busca información específica.
    
    Args:
        nombre: Qué buscar
    
    Returns:
        Información encontrada
    """

# 3. Names en snake_case
@tool("buscar_ley")  # ✅

# 4. Temperature baja
ChatGoogleGenerativeAI(temperature=0.3)  # ✅

# 5. Field con descripciones
campo: str = Field(description="Descripción clara")  # ✅
```

### ❌ DON'T (No hagas esto)

```python
# 1. Sin type hints
def buscar(nombre):  # ❌

# 2. Docstring vago
"""Busca algo"""  # ❌

# 3. Nombres con espacios
@tool("Buscar Ley")  # ❌

# 4. Temperature alta para tools
ChatGoogleGenerativeAI(temperature=1.5)  # ❌

# 5. Sin validación
campo: str  # Sin Field()  # ❌
```

---

## 🧪 Ejemplos de Código

### Ejemplo 1: Herramienta Básica

```python
from langchain.tools import tool

@tool
def buscar_ley(nombre: str) -> str:
    """Busca información sobre una ley colombiana."""
    leyes = {
        "tutela": "Acción de Tutela - Art. 86 CP",
        "habeas_corpus": "Habeas Corpus - Art. 30 CP"
    }
    return leyes.get(nombre, "Ley no encontrada")
```

### Ejemplo 2: Schema Complejo

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class CompararInput(BaseModel):
    ley1: str = Field(description="Primera ley")
    ley2: str = Field(description="Segunda ley")
    incluir_tiempos: bool = Field(default=True)

@tool(args_schema=CompararInput)
def comparar(ley1: str, ley2: str, incluir_tiempos: bool = True) -> str:
    """Compara dos leyes."""
    return f"Comparación: {ley1} vs {ley2}"
```

### Ejemplo 3: ToolNode

```python
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

tool_node = ToolNode([buscar_ley, comparar])

builder = StateGraph(MessagesState)
builder.add_node("agente", agente)
builder.add_node("herramientas", tool_node)
builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", condicion, {
    "herramientas": "herramientas",
    "fin": END
})
builder.add_edge("herramientas", "agente")
```

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangGraph ToolNode](https://docs.langchain.com/oss/python/langgraph/concepts/agentic_concepts#toolnode)
- [ToolRuntime](https://reference.langchain.com/python/langchain/tools/#langchain.tools.ToolRuntime)
- [Pydantic Output Parser](https://docs.langchain.com/oss/python/langchain/integrations/output_parsers/pydantic)

### Código de Ejemplo
- `src/course_examples/modulo_05/01_creacion_basica.py`
- `src/course_examples/modulo_05/02_schema_personalizado.py`
- `src/course_examples/modulo_05/03_toolnode_langgraph.py`

---

## 🎓 Próximos Pasos

### Módulo 6: Human in the Loop
- Breakpoints e interrupciones
- Human approval
- Edición de estado
- Time travel (replay)

### Módulo 7: RAG Fundamentos
- Document loaders
- Chunking strategies
- Embeddings
- Vector stores
- Retrieval básico

---

## 📝 Ejercicios Propuestos

### Ejercicio 1: Calculadora Legal
Crea herramientas para:
- Calcular intereses moratorios
- Convertir días hábiles a calendario
- Calcular fechas de vencimiento

### Ejercicio 2: Buscador de Jurisprudencia
Implementa:
- Búsqueda por palabra clave
- Filtro por corte/tribunal
- Filtro por fecha

### Ejercicio 3: Herramienta con Memoria
Crea una herramienta que:
- Guarde consultas frecuentes
- Recomiende basándose en historial
- Use `InMemoryStore` para persistencia

---

## ✅ Checklist de Completación

- [x] Leer documentación del README
- [x] Ejecutar `01_creacion_basica.py`
- [x] Ejecutar `02_schema_personalizado.py`
- [x] Ejecutar `03_toolnode_langgraph.py`
- [x] Completar ejercicios propuestos
- [ ] (Opcional) Crear herramienta personalizada
- [ ] (Opcional) Integrar con proyecto RAG

---

*Módulo completado: 2026-03-29*  
*Basado en documentación oficial de LangChain (Mayo 2025)*  
*Próximo módulo: Human in the Loop*
