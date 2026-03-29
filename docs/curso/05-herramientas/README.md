# Módulo 5: Herramientas (Tools) - Guía Completa

> **Nota**: Este módulo fue creado con información de la documentación oficial de LangChain (Mayo 2025) y probado con Google Gemini API.

---

## 📋 Índice del Módulo

1. [5.1 - Creación Básica de Herramientas](#51-creación-básica-de-herramientas)
2. [5.2 - Herramientas con Schema Personalizado](#52-herramientas-con-schema-personalizado)
3. [5.3 - ToolNode en LangGraph](#53-toolnode-en-langgraph)
4. [5.4 - Herramientas con Estado y Contexto](#54-herramientas-con-estado-y-contexto)
5. [5.5 - Herramientas con Streaming](#55-herramientas-con-streaming)
6. [5.6 - Comparativa: LangChain vs LangGraph vs LlamaIndex](#56-comparativa-langchain-vs-langgraph-vs-llamaindex)

---

## 5.1 - Creación Básica de Herramientas

### ¿Qué es una Herramienta?

Una **herramienta** es una función que tu agente puede llamar para interactuar con sistemas externos. Las herramientas extienden las capacidades del LLM más allá de su conocimiento de entrenamiento.

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTE CON TOOLS                     │
├─────────────────────────────────────────────────────────┤
│  1. Usuario hace pregunta                               │
│  2. LLM decide si llamar herramienta                    │
│  3. Herramienta ejecuta función                         │
│  4. Resultado vuelve al LLM                             │
│  5. LLM genera respuesta final                          │
└─────────────────────────────────────────────────────────┘
```

### Método 1: Decorador @tool (Recomendado)

```python
from langchain.tools import tool

@tool
def buscar_ley(nombre: str) -> str:
    """
    Busca información sobre una ley o mecanismo legal colombiano.
    
    Args:
        nombre: Nombre de la ley (ej: 'tutela', 'derecho_peticion')
    
    Returns:
        Información completa de la ley
    """
    leyes = {
        "tutela": {
            "nombre": "Acción de Tutela",
            "descripcion": "Mecanismo para proteger derechos fundamentales",
            "articulo": "Artículo 86 Constitución",
            "tiempo": "10 días hábiles"
        },
        "derecho_peticion": {
            "nombre": "Derecho de Petición",
            "descripcion": "Derecho para solicitar información",
            "articulo": "Artículo 23 Constitución",
            "tiempo": "15 días hábiles"
        }
    }
    
    nombre_key = nombre.lower().replace(" ", "_")
    
    if nombre_key in leyes:
        ley = leyes[nombre_key]
        return f"""
        Ley: {ley['nombre']}
        Descripción: {ley['descripcion']}
        Base Legal: {ley['articulo']}
        Tiempo de respuesta: {ley['tiempo']}
        """
    else:
        return f"Ley '{nombre}' no encontrada. Opciones: {', '.join(leyes.keys())}"
```

### Propiedades Automáticas

LangChain extrae automáticamente:

```python
print(f"Nombre: {buscar_ley.name}")
# Nombre: buscar_ley

print(f"Descripción: {buscar_ley.description}")
# Descripción: Busca información sobre una ley...

print(f"Schema: {buscar_ley.args}")
# Schema: {'nombre': {'title': 'Nombre', 'type': 'string'}}
```

### Método 2: Personalizar Nombre y Descripción

```python
@tool("buscar_normativa", 
      description="Busca normas y leyes colombianas. Úsala cuando el usuario pregunte sobre mecanismos legales.")
def buscar_ley_personalizado(nombre: str) -> str:
    """Docstring interno (no se usa como descripción)"""
    return "Resultado..."
```

### Errores Comunes

```python
# ❌ MAL: Sin type hints
@tool
def buscar(nombre):  # Sin tipo de dato
    return "algo"

# ✅ BIEN: Con type hints
@tool
def buscar(nombre: str) -> str:
    return "algo"

# ❌ MAL: Docstring vago
@tool
def buscar(nombre: str) -> str:
    """Busca algo"""  # Muy vago
    return "algo"

# ✅ BIEN: Docstring descriptivo
@tool
def buscar(nombre: str) -> str:
    """
    Busca información específica.
    
    Args:
        nombre: Qué buscar
        
    Returns:
        Información encontrada
    """
    return "algo"
```

---

## 5.2 - Herramientas con Schema Personalizado

### Usando Pydantic para Schemas Complejos

Cuando necesitas múltiples argumentos o validación avanzada:

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

# 1. Define el schema con Pydantic
class CompararLeyesInput(BaseModel):
    ley1: str = Field(
        description="Primera ley a comparar (ej: 'tutela')"
    )
    ley2: str = Field(
        description="Segunda ley a comparar (ej: 'habeas_corpus')"
    )
    incluir_tiempos: bool = Field(
        default=True,
        description="Si incluir comparación de tiempos de respuesta"
    )

# 2. Crea la herramienta con el schema
@tool(args_schema=CompararLeyesInput)
def comparar_leyes(
    ley1: str, 
    ley2: str, 
    incluir_tiempos: bool = True
) -> str:
    """Compara dos leyes colombianas y muestra sus diferencias y similitudes."""
    
    leyes = {
        "tutela": {"nombre": "Tutela", "tiempo": "10 días"},
        "derecho_peticion": {"nombre": "Derecho de Petición", "tiempo": "15 días"},
        "habeas_corpus": {"nombre": "Habeas Corpus", "tiempo": "36 horas"}
    }
    
    if ley1 not in leyes or ley2 not in leyes:
        return "Una o ambas leyes no existen"
    
    resultado = f"""
    COMPARACIÓN: {leyes[ley1]['nombre']} vs {leyes[ley2]['nombre']}
    
    Similitudes:
    - Ambas son mecanismos constitucionales
    - Ambas protegen derechos fundamentales
    
    Diferencias:
    """
    
    if incluir_tiempos:
        resultado += f"""
    Tiempos de respuesta:
    - {leyes[ley1]['nombre']}: {leyes[ley1]['tiempo']}
    - {leyes[ley2]['nombre']}: {leyes[ley2]['tiempo']}
    """
    
    return resultado
```

### Validación Automática

Pydantic valida automáticamente:

```python
# ✅ Válido
comparar_leyes.invoke({
    "ley1": "tutela",
    "ley2": "habeas_corpus",
    "incluir_tiempos": True
})

# ❌ Invalido - Pydantic rechazará
comparar_leyes.invoke({
    "ley1": 123,  # Debería ser string
    "ley2": "habeas_corpus"
})
# ValidationError: str type expected
```

### Schema con Literales (Valores Fijos)

```python
from typing import Literal

class CalcularFechaInput(BaseModel):
    dias: int = Field(description="Número de días")
    tipo: Literal["habiles", "calendario"] = Field(
        description="Tipo de días a calcular"
    )

@tool(args_schema=CalcularFechaInput)
def calcular_fecha(dias: int, tipo: str) -> str:
    """Calcula fechas en derecho procesal."""
    from datetime import datetime, timedelta
    
    hoy = datetime.now()
    
    if tipo == "habiles":
        # Simplificación: solo suma días (en producción, excluir fines de semana)
        resultado = hoy + timedelta(days=dias)
    else:
        resultado = hoy + timedelta(days=dias)
    
    return f"Hoy: {hoy.strftime('%Y-%m-%d')} → {dias} días {tipo}: {resultado.strftime('%Y-%m-%d')}"
```

---

## 5.3 - ToolNode en LangGraph

### ¿Qué es ToolNode?

**ToolNode** es un nodo pre-construido de LangGraph que ejecuta herramientas automáticamente cuando el LLM las llama.

### Grafo con ToolNode

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define herramientas
@tool
def buscar_ley(nombre: str) -> str:
    """Busca información sobre una ley colombiana."""
    return f"Información de {nombre}"

@tool
def calcular_fecha(dias: int, tipo: str = "habiles") -> str:
    """Calcula fechas procesales."""
    return f"Fecha calculada: {dias} días"

# 2. Crea ToolNode
tool_node = ToolNode([buscar_ley, calcular_fecha])

# 3. Prepara el LLM con herramientas
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
).bind_tools([buscar_ley, calcular_fecha])

# 4. Define nodo del agente
def agente(state: MessagesState):
    """El agente decide si llamar herramientas"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 5. Condición: ¿Hay tool_calls?
def debe_ejecutar_herramientas(state: MessagesState) -> str:
    ultimo_mensaje = state["messages"][-1]
    
    # Verifica si el LLM llamó herramientas
    if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
        return "herramientas"
    return "fin"

# 6. Construye el grafo
builder = StateGraph(MessagesState)

# Agrega nodos
builder.add_node("agente", agente)
builder.add_node("herramientas", tool_node)

# Agrega edges
builder.add_edge(START, "agente")
builder.add_conditional_edges(
    "agente",
    debe_ejecutar_herramientas,
    {
        "herramientas": "herramientas",
        "fin": END
    }
)

# Después de herramientas, vuelve al agente
builder.add_edge("herramientas", "agente")

# Compila
graph = builder.compile()

# 7. Usa el grafo
response = graph.invoke({
    "messages": [HumanMessage(content="¿Qué es una tutela?")]
})

print(response["messages"][-1].content)
```

### Ventajas de ToolNode

| Ventaja | Descripción |
|---------|-------------|
| ✅ **Manejo automático** | Ejecuta herramientas en paralelo |
| ✅ **Error handling** | Maneja errores de herramientas |
| ✅ **State injection** | Inyecta resultados en el estado |
| ✅ **ToolMessage** | Crea mensajes de herramienta automáticamente |

---

## 5.4 - Herramientas con Estado y Contexto

### Acceder al Estado (State)

Las herramientas pueden leer y modificar el estado de la conversación:

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

# Herramienta que LEE estado
@tool
def obtener_preferencias(runtime: ToolRuntime) -> str:
    """Obtiene las preferencias guardadas del usuario."""
    
    preferencias = runtime.state.get("user_preferences", {})
    
    if not preferencias:
        return "No hay preferencias guardadas"
    
    return ", ".join(f"{k}={v}" for k, v in preferencias.items())

# Herramienta que ESCRIBE estado
@tool
def guardar_preferencia(clave: str, valor: str) -> Command:
    """Guarda una preferencia del usuario."""
    
    return Command(
        update={
            "user_preferences": {clave: valor}
        }
    )
```

### Usar Contexto (Context)

El contexto es información inmutable pasada en la invocación:

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define el schema de contexto
@dataclass
class UserContext:
    user_id: str
    rol: str
    especialidad: str

# 2. Herramienta que usa contexto
@tool
def obtener_info_usuario(runtime: ToolRuntime) -> str:
    """Obtiene información del usuario actual."""
    
    user_id = runtime.context.user_id
    rol = runtime.context.rol
    especialidad = runtime.context.especialidad
    
    return f"Usuario: {user_id}, Rol: {rol}, Especialidad: {especialidad}"

# 3. Crea agente con contexto
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

agent = create_agent(
    llm,
    tools=[obtener_info_usuario],
    context_schema=UserContext,
)

# 4. Invoca con contexto
response = agent.invoke(
    {"messages": [{"role": "user", "content": "¿Quién soy yo?"}]},
    context=UserContext(
        user_id="usuario_123",
        rol="abogado",
        especialidad="derecho_laboral"
    )
)
```

### Memoria a Largo Plazo (Store)

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

# 1. Crea store persistente
store = InMemoryStore()

# 2. Herramientas que usan store
@tool
def guardar_info_usuario(
    user_id: str, 
    info: dict[str, Any], 
    runtime: ToolRuntime
) -> str:
    """Guarda información del usuario en memoria persistente."""
    
    store = runtime.store
    store.put(("users",), user_id, info)
    
    return f"Información guardada para usuario {user_id}"

@tool
def obtener_info_usuario(user_id: str, runtime: ToolRuntime) -> str:
    """Obtiene información guardada de un usuario."""
    
    store = runtime.store
    info = store.get(("users",), user_id)
    
    if info:
        return str(info.value)
    return "Usuario no encontrado"

# 3. Crea agente con store
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

agent = create_agent(
    llm,
    tools=[guardar_info_usuario, obtener_info_usuario],
    store=store
)
```

---

## 5.5 - Herramientas con Streaming

### Stream Writer

Emite updates en tiempo real durante la ejecución:

```python
from langchain.tools import tool, ToolRuntime
import time

@tool
def proceso_largo(cantidad: int, runtime: ToolRuntime) -> str:
    """Simula un proceso largo con actualizaciones en tiempo real."""
    
    writer = runtime.stream_writer
    
    # Emite updates
    writer(f"🚀 Iniciando proceso de {cantidad} pasos...")
    
    for i in range(cantidad):
        time.sleep(0.5)  # Simula trabajo
        writer(f"⚙️ Paso {i+1}/{cantidad} completado")
    
    writer("✅ Proceso completado")
    
    return f"Proceso completado: {cantidad} pasos"
```

### Consumir Streaming

```python
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
agent = create_agent(llm, tools=[proceso_largo])

# Stream de eventos
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Ejecuta 5 pasos"}]
}):
    if "messages" in chunk:
        for msg in chunk["messages"]:
            if hasattr(msg, "content") and msg.content:
                print(f"  {msg.content}")
```

---

## 5.6 - Comparativa: LangChain vs LangGraph vs LlamaIndex

### Tabla Comparativa

| Característica | LangChain | LangGraph | LlamaIndex |
|----------------|-----------|-----------|------------|
| **Creación** | `@tool` | `@tool` + `ToolNode` | `FunctionTool.from_defaults()` |
| **Schema** | Pydantic | Pydantic | Pydantic + Annotated |
| **Estado** | `ToolRuntime.state` | State injection | Limitado |
| **Contexto** | `ToolRuntime.context` | Context schema | No soportado |
| **Store** | `ToolRuntime.store` | BaseStore | Return direct |
| **Streaming** | `stream_writer` | Vía ToolNode | No soportado |
| **Error Handling** | Avanzado | ToolNode maneja | Básico |
| **Use Case** | Agentes simples | Agentes complejos | RAG + queries |

### Ejemplo: Misma Herramienta en los 3 Frameworks

#### LangChain

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def buscar(nombre: str) -> str:
    """Busca una ley."""
    return f"Ley: {nombre}"

agent = create_agent(llm, tools=[buscar])
```

#### LangGraph

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState

@tool
def buscar(nombre: str) -> str:
    """Busca una ley."""
    return f"Ley: {nombre}"

tool_node = ToolNode([buscar])
# ... construir grafo con ToolNode
```

#### LlamaIndex

```python
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent

def buscar(nombre: str) -> str:
    """Busca una ley."""
    return f"Ley: {nombre}"

tool = FunctionTool.from_defaults(buscar)
agent = FunctionAgent(llm=llm, tools=[tool])
```

### Cuándo Usar Cada Uno

```
┌─────────────────────────────────────────────────────────┐
│  ¿Qué framework elegir?                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  LangChain → Agente simple con tools                    │
│  - create_agent es suficiente                           │
│  - No necesitas control total del flujo                 │
│                                                         │
│  LangGraph → Agente complejo                            │
│  - Necesitas control del flujo                          │
│  - Múltiples herramientas en paralelo                   │
│  - Estado personalizado                                 │
│  - Human-in-the-loop                                    │
│                                                         │
│  LlamaIndex → RAG + queries                             │
│  - Tu caso de uso es principalmente RAG                 │
│  - Queries sobre documentos                             │
│  - Ya usas LlamaIndex para embeddings                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Ejercicios Prácticos

### Ejercicio 1: Calculadora Legal

Crea herramientas para:
1. Calcular intereses moratorios
2. Convertir días hábiles a calendario
3. Calcular fechas de vencimiento

### Ejercicio 2: Buscador de Jurisprudencia

Implementa:
1. Búsqueda por palabra clave
2. Filtro por corte/tribunal
3. Filtro por fecha

### Ejercicio 3: Herramienta con Memoria

Crea una herramienta que:
1. Guarde consultas frecuentes
2. Recomiende basándose en historial
3. Use `InMemoryStore` para persistencia

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangGraph ToolNode](https://docs.langchain.com/oss/python/langgraph/concepts/agentic_concepts#toolnode)
- [ToolRuntime](https://reference.langchain.com/python/langchain/tools/#langchain.tools.ToolRuntime)

### Siguiente Módulo
➡️ **Módulo 6: Human in the Loop**

---

*Módulo creado: 2026-03-29*
*Basado en documentación oficial de LangChain (Mayo 2025)*
