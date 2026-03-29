"""
Módulo 6 - Human in the Loop (HITL)

Objetivo: Aprender a usar interrupts para aprobación humana
Basado en: https://docs.langchain.com/oss/python/langgraph/interrupts
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage

# Cargar variables de entorno
load_dotenv()


# ============================================================================
# EJEMPLO 1: Aprobación Simple
# ============================================================================

class AprobacionState(TypedDict):
    """Estado para flujo de aprobación."""
    accion: str
    detalles: str
    aprobado: Optional[bool]
    estado: Literal["pendiente", "aprobado", "rechazado"]


def nodo_aprobacion_simple(state: AprobacionState) -> Command[Literal["aprobar", "rechazar"]]:
    """
    Nodo que pausa para aprobación humana.
    
    Args:
        state: Estado actual con detalles de la acción
    
    Returns:
        Command para routear a aprobar o rechazar
    """
    
    # Pausar y pedir aprobación
    decision = interrupt({
        "tipo": "aprobacion_simple",
        "pregunta": "¿Apruebas esta acción?",
        "accion": state["accion"],
        "detalles": state["detalles"]
    })
    
    # Routear basado en decisión
    if decision:
        return Command(goto="aprobar")
    else:
        return Command(goto="rechazar")


def nodo_aprobado(state: AprobacionState):
    """Acción aprobada."""
    print(f"✅ Acción aprobada: {state['accion']}")
    return {"estado": "aprobado", "aprobado": True}


def nodo_rechazado(state: AprobacionState):
    """Acción rechazada."""
    print(f"❌ Acción rechazada: {state['accion']}")
    return {"estado": "rechazado", "aprobado": False}


def crear_grafo_aprobacion_simple():
    """
    Crea grafo de aprobación simple.
    
    Returns:
        Grafo compilado con checkpointer
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 1: APROBACIÓN SIMPLE")
    print("=" * 80)
    
    builder = StateGraph(AprobacionState)
    
    # Nodos
    builder.add_node("revision", nodo_aprobacion_simple)
    builder.add_node("aprobar", nodo_aprobado)
    builder.add_node("rechazar", nodo_rechazado)
    
    # Edges
    builder.add_edge(START, "revision")
    builder.add_edge("aprobar", END)
    builder.add_edge("rechazar", END)
    
    # Compilar con checkpointer
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    print("\n✅ Grafo de aprobación simple creado")
    print("   Estructura: START → revision → [aprobar|rechazar] → END")
    
    return graph


# ============================================================================
# EJEMPLO 2: Revisión y Edición de Documento
# ============================================================================

class DocumentoState(TypedDict):
    """Estado para revisión de documentos."""
    titulo: str
    contenido: str
    version: int
    revisiones: List[str]
    finalizado: bool


def nodo_generar_borrador(state: DocumentoState):
    """
    Genera borrador inicial (simula LLM).
    
    Args:
        state: Estado con título del documento
    
    Returns:
        Estado con borrador generado
    """
    print("\n📝 Generando borrador...")
    
    # Simulación de generación con LLM
    borrador = f"""
    {state['titulo'].upper()}
    
    CAPÍTULO I - DISPOSICIONES GENERALES
    
    Artículo 1°. Objeto: El presente documento establece...
    
    Artículo 2°. Ámbito de aplicación: Las disposiciones contenidas...
    
    CAPÍTULO II - DERECHOS Y OBLIGACIONES
    
    Artículo 3°. Derechos: Los beneficiarios tendrán derecho a...
    
    Artículo 4°. Obligaciones: Son obligaciones de las partes...
    
    DISPOSICIONES FINALES
    
    Artículo 5°. Vigencia: El presente documento entrará en vigor...
    """
    
    return {
        "contenido": borrador,
        "version": 1,
        "revisiones": ["Borrador inicial generado"]
    }


def nodo_revision_documento(state: DocumentoState):
    """
    Pausa para revisión y edición humana del documento.
    
    Args:
        state: Estado con borrador
    
    Returns:
        Estado con documento revisado
    """
    print("\n👀 Revisión de documento requerida...")
    
    # Pausar para revisión
    contenido_revisado = interrupt({
        "tipo": "revision_documento",
        "instruccion": "Revisa y edita el borrador del documento",
        "documento": {
            "titulo": state["titulo"],
            "contenido": state["contenido"],
            "version": state["version"]
        },
        "checklist": [
            "✓ Terminología legal correcta",
            "✓ Referencias a leyes vigentes",
            "✓ Redacción clara y precisa",
            "✓ Sin ambigüedades",
            "✓ Formato apropiado"
        ]
    })
    
    return {
        "contenido": contenido_revisado,
        "version": state["version"] + 1,
        "revisiones": state["revisiones"] + ["Revisión humana completada"],
        "finalizado": True
    }


def crear_grafo_revision_documento():
    """
    Crea grafo de revisión de documentos.
    
    Returns:
        Grafo compilado con checkpointer
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 2: REVISIÓN DE DOCUMENTO LEGAL")
    print("=" * 80)
    
    builder = StateGraph(DocumentoState)
    
    # Nodos
    builder.add_node("generar", nodo_generar_borrador)
    builder.add_node("revisar", nodo_revision_documento)
    
    # Edges
    builder.add_edge(START, "generar")
    builder.add_edge("generar", "revisar")
    builder.add_edge("revisar", END)
    
    # Compilar
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    print("\n✅ Grafo de revisión de documentos creado")
    print("   Estructura: START → generar → revisar → END")
    
    return graph


# ============================================================================
# EJEMPLO 3: Herramienta con Aprobación (Tool + Interrupt)
# ============================================================================

from langchain.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def enviar_comunicado_oficial(destinatario: str, asunto: str, contenido: str):
    """
    Envía un comunicado oficial a un destinatario.
    
    Esta herramienta requiere aprobación antes de enviar.
    
    Args:
        destinatario: Email o dirección del destinatario
        asunto: Asunto del comunicado
        contenido: Contenido del mensaje
    """
    
    print(f"\n📧 Preparando envío de comunicado...")
    
    # Pausar para aprobación
    aprobacion = interrupt({
        "tipo": "aprobacion_envio",
        "accion": "enviar_comunicado_oficial",
        "pregunta": "¿Aprobar envío de este comunicado?",
        "detalles": {
            "destinatario": destinatario,
            "asunto": asunto,
            "contenido_preview": contenido[:200] + "..." if len(contenido) > 200 else contenido
        },
        "advertencia": "Este envío es oficial y queda registrado"
    })
    
    # Procesar decisión
    if aprobacion.get("aprobar"):
        # Posiblemente con modificaciones
        final_destinatario = aprobacion.get("destinatario", destinatario)
        final_asunto = aprobacion.get("asunto", asunto)
        final_contenido = aprobacion.get("contenido", contenido)
        
        # Simular envío
        print(f"✅ Comunicado enviado a {final_destinatario}")
        print(f"   Asunto: {final_asunto}")
        
        return f"Comunicado oficial enviado a {final_destinatario} con asunto '{final_asunto}'"
    else:
        print("❌ Envío cancelado por usuario")
        return "Envío cancelado por aprobación denegada"


@tool
def consultar_normativa(tema: str):
    """
    Consulta normativa legal sobre un tema.
    
    Args:
        tema: Tema de consulta legal
    """
    
    print(f"\n📚 Consultando normativa sobre: {tema}")
    
    # Simulación de consulta
    normativas = {
        "tutela": "Constitución Política, Artículo 86. Decreto 2591 de 1991",
        "derecho_peticion": "Constitución Política, Artículo 23. Ley 1755 de 2015",
        "habeas_corpus": "Constitución Política, Artículo 30. Ley 1095 de 2006"
    }
    
    return normativas.get(
        tema.lower(),
        f"No se encontró normativa específica para '{tema}'"
    )


def crear_grafo_con_herramientas():
    """
    Crea grafo con herramientas que requieren aprobación.
    
    Returns:
        Grafo compilado con checkpointer
    """
    print("\n" + "=" * 80)
    print("EJEMPLO 3: HERRAMIENTAS CON APROBACIÓN")
    print("=" * 80)
    
    # ToolNode con herramientas
    tool_node = ToolNode([enviar_comunicado_oficial, consultar_normativa])
    
    # Grafo simple
    builder = StateGraph(MessagesState)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    
    # Compilar
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    print("\n✅ Grafo con herramientas creado")
    print("   Herramientas: enviar_comunicado_oficial, consultar_normativa")
    print("   enviar_comunicado_oficial requiere aprobación humana")
    
    return graph


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def demo_aprobacion_simple():
    """Demuestra flujo de aprobación simple."""
    
    print("\n" + "═" * 80)
    print("DEMOSTRACIÓN 1: APROBACIÓN SIMPLE")
    print("═" * 80)
    
    graph = crear_grafo_aprobacion_simple()
    config = {"configurable": {"thread_id": "aprobacion-demo-001"}}
    
    # Invocar inicial
    print("\n📋 Invocación inicial (se pausará para aprobación)...")
    resultado = graph.invoke({
        "accion": "Enviar demanda laboral",
        "detalles": "Demanda por despido injustificado contra Empresa XYZ",
        "aprobado": None,
        "estado": "pendiente"
    }, config)
    
    print(f"\n⏸️  Grafo pausado en interrupt()")
    print(f"Interrupt payload: {resultado['__interrupt__'][0].value}")
    
    # Reanudar con aprobación
    print("\n▶️  Reanudando con aprobación (True)...")
    resultado_final = graph.invoke(Command(resume=True), config)
    
    print(f"\n✅ Resultado final: {resultado_final}")


def demo_revision_documento():
    """Demuestra revisión de documento."""
    
    print("\n" + "═" * 80)
    print("DEMOSTRACIÓN 2: REVISIÓN DE DOCUMENTO")
    print("═" * 80)
    
    graph = crear_grafo_revision_documento()
    config = {"configurable": {"thread_id": "documento-demo-001"}}
    
    # Invocar inicial
    print("\n📝 Generando borrador (se pausará para revisión)...")
    resultado = graph.invoke({
        "titulo": "CONTRATO DE PRESTACIÓN DE SERVICIOS PROFESIONALES",
        "contenido": "",
        "version": 0,
        "revisiones": [],
        "finalizado": False
    }, config)
    
    print(f"\n⏸️  Grafo pausado en interrupt()")
    interrupt_info = resultado['__interrupt__'][0].value
    print(f"Tipo: {interrupt_info['tipo']}")
    print(f"Instrucción: {interrupt_info['instruccion']}")
    print(f"Checklist: {len(interrupt_info['checklist'])} items")
    
    # Simular revisión humana
    contenido_editado = resultado['__interrupt__'][0].value.copy()
    contenido_editado['documento']['contenido'] += "\n\n[EDITADO POR HUMANO: Se agregaron cláusulas adicionales]"
    
    print("\n▶️  Reanudando con contenido editado...")
    # En producción: graph.invoke(Command(resume=contenido_editado), config)


def demo_herramientas_con_aprobacion():
    """Demuestra herramientas con aprobación."""
    
    print("\n" + "═" * 80)
    print("DEMOSTRACIÓN 3: HERRAMIENTAS CON APROBACIÓN")
    print("═" * 80)
    
    graph = crear_grafo_con_herramientas()
    config = {"configurable": {"thread_id": "tools-demo-001"}}
    
    # Consultar normativa (no requiere aprobación)
    print("\n📚 Consulta 1: Consultar normativa (sin aprobación)")
    resultado = graph.invoke({
        "messages": [HumanMessage(content="Consulta sobre tutela")]
    }, config)
    print(f"✅ Resultado: {resultado['messages'][-1].content}")
    
    # Enviar comunicado (requiere aprobación)
    print("\n📧 Consulta 2: Enviar comunicado (requiere aprobación)")
    resultado = graph.invoke({
        "messages": [HumanMessage(content="Envía comunicado a juez sobre caso 123")]
    }, config)
    
    if '__interrupt__' in resultado:
        print(f"\n⏸️  Herramienta pausada para aprobación")
        interrupt_info = resultado['__interrupt__'][0].value
        print(f"Pregunta: {interrupt_info['pregunta']}")
        print(f"Advertencia: {interrupt_info['advertencia']}")
        
        print("\n▶️  Para aprobar: graph.invoke(Command(resume={'aprobar': True}), config)")


def main():
    """Función principal de demostración."""
    
    print("=" * 80)
    print("MÓDULO 6 - HUMAN IN THE LOOP (HITL)")
    print("=" * 80)
    print("\nEste módulo demuestra cómo usar interrupts para:")
    print("1. Aprobación humana de acciones")
    print("2. Revisión y edición de contenido")
    print("3. Herramientas que requieren aprobación")
    
    # Ejecutar demostraciones
    demo_aprobacion_simple()
    demo_revision_documento()
    demo_herramientas_con_aprobacion()
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
    print("\n💡 Nota: Los interrupts requieren checkpointer para guardar estado.")
    print("   En producción, usa un checkpointer persistente (PostgreSQL, Redis, etc.)")
    print("\n📚 Para más información:")
    print("   https://docs.langchain.com/oss/python/langgraph/interrupts")


# ============================================================================
# MEJORES PRÁCTICAS
# ============================================================================

"""
✅ MEJORES PRÁCTICAS PARA HUMAN IN THE LOOP:

1. Usa checkpointer persistente en producción
   checkpointer = PostgresSaver(...)  # ✅
   checkpointer = MemorySaver()       # ⚠️ Solo desarrollo

2. Proporciona contexto claro en el interrupt
   interrupt({
       "pregunta": "¿Qué quieres hacer?",
       "opciones": ["aprobar", "rechazar", "editar"],
       "contexto": state["datos"]
   })

3. Maneja múltiples interrupts con IDs
   resume_map = {id: valor for id, valor in interrupts}

4. Valida entrada humana antes de continuar
   if not validar_entrada(humano_input):
       return Command(goto="solicitar_nuevamente")

5. No envuelvas interrupt() en try/except
   # ❌ MAL
   try:
       valor = interrupt()
   except Exception:
       ...
   
   # ✅ BIEN
   valor = interrupt()  # Sin try/except


❌ ERRORES COMUNES:

1. Olvidar el checkpointer
   graph = builder.compile()  # ❌ Sin checkpointer
   # interrupt() no funcionará

2. Usar thread_id incorrecto al reanudar
   graph.invoke(Command(resume=valor), config={"thread_id": "otro"})  # ❌

3. No proporcionar suficiente contexto
   interrupt("¿Apruebas?")  # ❌ Muy vago
   
   interrupt({
       "accion": "...",
       "detalles": "...",
       "consecuencias": "..."
   })  # ✅
"""


if __name__ == "__main__":
    main()
