# Módulo 6: Human in the Loop (HITL)

> **Basado en**: [Documentación Oficial de LangGraph - Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)  
> **Estado**: ✅ En Desarrollo  
> **Prerrequisitos**: Módulos 1-5 completados

---

## 📋 Índice del Módulo

1. [6.1 - Introducción a Interrupts](#61-introducción-a-interrupts)
2. [6.2 - Aprobación o Rechazo](#62-aprobación-o-rechazo)
3. [6.3 - Revisión y Edición de Estado](#63-revisión-y-edición-de-estado)
4. [6.4 - Interrupts en Herramientas](#64-interrupts-en-herramientas)
5. [6.5 - Múltiples Interrupts](#65-múltiples-interrupts)
6. [6.6 - Time Travel (Viaje en el Tiempo)](#66-time-travel-viaje-en-el-tiempo)

---

## 6.1 - Introducción a Interrupts

### ¿Qué es Human in the Loop (HITL)?

**Human in the Loop** permite pausar la ejecución del grafo en puntos específicos y esperar entrada externa antes de continuar. Esto es esencial para:

- ✅ Aprobación de acciones críticas
- ✅ Revisión de contenido generado por LLM
- ✅ Validación de decisiones del agente
- ✅ Edición manual de resultados

### ¿Cómo Funciona `interrupt()`?

```
┌─────────────────────────────────────────────────────────┐
│                 FLUJO CON INTERRUPT                     │
├─────────────────────────────────────────────────────────┤
│  1. Grafo ejecuta nodo → llama a interrupt()            │
│  2. LangGraph guarda estado (checkpoint)                │
│  3. Ejecución se pausa indefinidamente                  │
│  4. Humano provee entrada                               │
│  5. Grafo se reanuda con Command(resume=valor)          │
│  6. interrupt() retorna el valor proporcionado          │
└─────────────────────────────────────────────────────────┘
```

### Requisitos para Usar Interrupts

```python
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

# 1. Checkpointer para guardar estado
checkpointer = MemorySaver()

# 2. Thread ID para identificar el estado
config = {"configurable": {"thread_id": "mi-hilo-123"}}

# 3. Llamar interrupt() donde pausar
def mi_nodo(state):
    valor = interrupt("¿Continuar?")
    return {"resultado": valor}
```

### Ejemplo Básico

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    input_data: str
    aprobado: bool

def nodo_aprobacion(state: State):
    # Pausar y pedir aprobación
    aprobado = interrupt({
        "pregunta": "¿Apruebas esta acción?",
        "datos": state["input_data"]
    })
    
    # Cuando se reanuda, aprobado contiene el valor de Command(resume=...)
    return {"aprobado": aprobado}

# Crear grafo
builder = StateGraph(State)
builder.add_node("aprobacion", nodo_aprobacion)
builder.add_edge(START, "aprobacion")
builder.add_edge("aprobacion", END)

# Compilar con checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Uso
config = {"configurable": {"thread_id": "aprobacion-001"}}

# Primera invocación → se pausa en interrupt()
resultado = graph.invoke({"input_data": "datos importantes"}, config)
print(resultado["__interrupt__"])  # [Interrupt(value={'pregunta': ...})]

# Reanudar con decisión humana
graph.invoke(Command(resume=True), config)  # True → aprobado
```

---

## 6.2 - Aprobación o Rechazo

### Patrón: Approval Workflow

Uno de los usos más comunes es pausar antes de acciones críticas:

```python
from typing import Literal
from langgraph.types import interrupt, Command

def nodo_aprobacion(state: State) -> Command[Literal["proceder", "cancelar"]]:
    """
    Pausa antes de ejecutar acción crítica.
    """
    
    decision = interrupt({
        "pregunta": "¿Aprobar esta acción?",
        "detalles": state["accion_detalle"],
        "riesgo": state["riesgo_nivel"]
    })
    
    # Routear basado en decisión
    if decision:
        return Command(goto="proceder")
    else:
        return Command(goto="cancelar")
```

### Ejemplo Completo: Aprobación de Email

```python
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

class EmailState(TypedDict):
    destinatario: str
    asunto: str
    cuerpo: str
    estado: Literal["pendiente", "aprobado", "rechazado"]
    enviado: bool

def nodo_revision(state: EmailState) -> Command[Literal["enviar", "cancelar"]]:
    """
    Revisa email antes de enviar.
    """
    
    decision = interrupt({
        "accion": "revisar_email",
        "pregunta": "¿Enviar este email?",
        "detalles": {
            "para": state["destinatario"],
            "asunto": state["asunto"],
            "cuerpo": state["cuerpo"][:100] + "..."  # Preview
        }
    })
    
    return Command(goto="enviar" if decision else "cancelar")

def nodo_enviar(state: EmailState):
    """Envía el email (simulado)."""
    print(f"✅ Email enviado a {state['destinatario']}")
    return {"enviado": True, "estado": "aprobado"}

def nodo_cancelar(state: EmailState):
    """Cancela el envío."""
    print("❌ Email cancelado")
    return {"enviado": False, "estado": "rechazado"}

# Construir grafo
builder = StateGraph(EmailState)
builder.add_node("revision", nodo_revision)
builder.add_node("enviar", nodo_enviar)
builder.add_node("cancelar", nodo_cancelar)

builder.add_edge(START, "revision")
builder.add_edge("enviar", END)
builder.add_edge("cancelar", END)

# Compilar
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Uso
config = {"configurable": {"thread_id": "email-001"}}

# Invocar inicial
resultado = graph.invoke({
    "destinatario": "cliente@empresa.com",
    "asunto": "Propuesta Comercial",
    "cuerpo": "Estimado cliente, adjuntamos propuesta...",
    "estado": "pendiente",
    "enviado": False
}, config)

# Resultado muestra interrupt
print(f"Interrupt: {resultado['__interrupt__']}")

# Reanudar con aprobación
graph.invoke(Command(resume=True), config)  # Aprobar
# O: graph.invoke(Command(resume=False), config)  # Rechazar
```

---

## 6.3 - Revisión y Edición de Estado

### Patrón: Review and Edit

Permite a humanos revisar y editar contenido generado por el LLM:

```python
from langgraph.types import interrupt

def nodo_revision(state: State):
    """
    Pausa para revisar y editar texto generado.
    """
    
    contenido_editado = interrupt({
        "instruccion": "Revisa y edita este contenido",
        "contenido_original": state["texto_generado"],
        "sugerencias": state.get("sugerencias", [])
    })
    
    # Actualizar estado con versión editada
    return {"texto_generado": contenido_editado}
```

### Ejemplo: Revisión de Documento Legal

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

class DocumentoLegal(TypedDict):
    titulo: str
    contenido: str
    version: int
    revisiones: List[str]

def nodo_generar_borrador(state: DocumentoLegal):
    """Genera borrador inicial con LLM."""
    # Simulación: en producción usarías LLM
    borrador = f"""
    {state['titulo']}
    
    Artículo 1: Disposiciones generales
    Artículo 2: Derechos y obligaciones
    Artículo 3: Vigencia
    """
    return {"contenido": borrador, "version": 1}

def nodo_revision_legal(state: DocumentoLegal):
    """
    Abogado revisa y edita el borrador.
    """
    
    contenido_revisado = interrupt({
        "instruccion": "Revisa el borrador del documento legal",
        "documento": {
            "titulo": state["titulo"],
            "contenido": state["contenido"],
            "version": state["version"]
        },
        "checklist": [
            "✓ Terminología legal correcta",
            "✓ Referencias a leyes actualizadas",
            "✓ Cláusulas completas",
            "✓ Sin ambigüedades"
        ]
    })
    
    return {
        "contenido": contenido_revisado,
        "version": state["version"] + 1,
        "revisiones": state.get("revisiones", []) + ["Revisión completada"]
    }

# Grafo
builder = StateGraph(DocumentoLegal)
builder.add_node("generar", nodo_generar_borrador)
builder.add_node("revisar", nodo_revision_legal)
builder.add_edge(START, "generar")
builder.add_edge("generar", "revisar")
builder.add_edge("revisar", END)

graph = builder.compile(checkpointer=MemorySaver())

# Uso
config = {"configurable": {"thread_id": "documento-001"}}

# Generar borrador
graph.invoke({
    "titulo": "CONTRATO DE PRESTACIÓN DE SERVICIOS",
    "contenido": "",
    "version": 0,
    "revisiones": []
}, config)

# Revisar (pausa para edición humana)
# graph.invoke(Command(resume="Contenido editado por abogado"), config)
```

---

## 6.4 - Interrupts en Herramientas

### Patrón: Tool Approval

Puedes usar `interrupt()` directamente dentro de herramientas:

```python
from langchain.tools import tool
from langgraph.types import interrupt

@tool
def enviar_email(destinatario: str, asunto: str, cuerpo: str):
    """
    Envía un email al destinatario especificado.
    
    Esta herramienta pausa para aprobación antes de enviar.
    """
    
    # Pausar para aprobación
    respuesta = interrupt({
        "accion": "enviar_email",
        "pregunta": "¿Aprobar envío de email?",
        "detalles": {
            "para": destinatario,
            "asunto": asunto,
            "cuerpo": cuerpo[:200]
        }
    })
    
    # Respuesta puede aprobar o modificar
    if respuesta.get("aprobar"):
        # Posiblemente con modificaciones
        final_destinatario = respuesta.get("para", destinatario)
        final_asunto = respuesta.get("asunto", asunto)
        final_cuerpo = respuesta.get("cuerpo", cuerpo)
        
        # Enviar email (simulado)
        print(f"Enviando email a {final_destinatario}...")
        return f"Email enviado a {final_destinatario}"
    else:
        return "Email cancelado por usuario"
```

### Ejemplo Completo: Herramienta con Aprobación

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def consultar_base_datos(query: str):
    """Consulta la base de datos legal."""
    # Simulación
    return f"Resultados para: {query}"

@tool
def ejecutar_accion_legal(accion: str, detalle: str):
    """
    Ejecuta una acción legal (demanda, tutela, etc.).
    
    Requiere aprobación antes de ejecutar.
    """
    
    # Pausar para aprobación
    aprobacion = interrupt({
        "tipo": "aprobacion_accion",
        "accion": accion,
        "detalle": detalle,
        "advertencia": "Esta acción es irreversible",
        "pregunta": "¿Ejecutar esta acción legal?"
    })
    
    if aprobacion.get("ejecutar"):
        # Ejecutar acción
        return f"✅ {accion} ejecutada exitosamente"
    else:
        return f"❌ {accion} cancelada"

# ToolNode
tool_node = ToolNode([consultar_base_datos, ejecutar_accion_legal])

# LLM con tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3
).bind_tools([consultar_base_datos, ejecutar_accion_legal])

def agente(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def debe_usar_tools(state: MessagesState) -> str:
    if hasattr(state["messages"][-1], "tool_calls"):
        return "tools"
    return "fin"

# Grafo
builder = StateGraph(MessagesState)
builder.add_node("agente", agente)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", debe_usar_tools, {
    "tools": "tools",
    "fin": END
})
builder.add_edge("tools", "agente")

graph = builder.compile(checkpointer=MemorySaver())

# Uso
config = {"configurable": {"thread_id": "accion-legal-001"}}

# El agente puede llamar ejecutar_accion_legal
# → Se pausará para aprobación
```

---

## 6.5 - Múltiples Interrupts

### Patrón: Parallel Branches

Cuando hay branches paralelos, pueden haber múltiples interrupts simultáneos:

```python
from typing import Annotated
import operator
from langgraph.types import interrupt, Command

class State(TypedDict):
    respuestas: Annotated[list, operator.add]

def nodo_a(state):
    respuesta_a = interrupt("Pregunta A")
    return {"respuestas": [f"A: {respuesta_a}"]}

def nodo_b(state):
    respuesta_b = interrupt("Pregunta B")
    return {"respuestas": [f"B: {respuesta_b}"]}

# Grafo con branches paralelos
builder = StateGraph(State)
builder.add_node("a", nodo_a)
builder.add_node("b", nodo_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")  # Paralelo
builder.add_edge("a", END)
builder.add_edge("b", END)

graph = builder.compile(checkpointer=MemorySaver())

# Uso
config = {"configurable": {"thread_id": "multiple-001"}}

# Ambos nodos se ejecutan en paralelo → 2 interrupts
resultado = graph.invoke({"respuestas": []}, config)

# resultado['__interrupt__'] tiene 2 interrupts
print(f"Interrupts: {len(resultado['__interrupt__'])}")

# Reanudar ambos con mapa de IDs
resume_map = {
    interrupt_info.id: f"Respuesta para {interrupt_info.value}"
    for interrupt_info in resultado['__interrupt__']
}

graph.invoke(Command(resume=resume_map), config)
```

---

## 6.6 - Time Travel (Viaje en el Tiempo)

### ¿Qué es Time Travel?

**Time Travel** permite volver a estados anteriores del grafo y re-ejecutar desde ese punto. Es útil para:

- ✅ Corregir errores
- ✅ Probar alternativas
- ✅ Depurar el flujo

### Cómo Funciona

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "viaje-001"}}

# Ejecutar grafo
graph.invoke({"input": "datos"}, config)

# Obtener historial de checkpoints
historial = checkpointer.list(config)

for checkpoint in historial:
    print(f"Checkpoint: {checkpoint.id}")
    print(f"  Estado: {checkpoint.state}")

# Volver a checkpoint específico
config_volver = {
    "configurable": {
        "thread_id": "viaje-001",
        "checkpoint_id": "checkpoint-especifico"
    }
}

# Re-ejecutar desde ese punto
graph.invoke(Command(resume="nueva_entrada"), config_volver)
```

---

## 🎯 Ejercicios Prácticos

### Ejercicio 1: Aprobación de Demanda

Crea un flujo que:
1. Genera borrador de demanda
2. Pausa para aprobación de abogado
3. Si aprueba → genera documento final
4. Si rechaza → vuelve a generar

### Ejercicio 2: Revisión de Contrato

Implementa:
1. LLM genera cláusulas
2. Humano revisa y edita cada cláusula
3. Sistema consolida versión final

### Ejercicio 3: Múltiples Aprobaciones

Crea un sistema que:
1. Envía a aprobación de 3 personas
2. Cada una revisa diferente sección
3. Requiere todas las aprobaciones para continuar

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Human in the Loop Patterns](https://docs.langchain.com/oss/python/langgraph/human_in_the_loop)
- [Time Travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)

### Siguiente Módulo
➡️ **Módulo 7: RAG Fundamentos**

---

*Módulo creado: 2026-03-29*  
*Basado en documentación oficial de LangGraph (Mayo 2025)*
