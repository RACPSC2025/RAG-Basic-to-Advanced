"""
Módulo 5.3 - ToolNode en LangGraph

Objetivo: Aprender a usar ToolNode para ejecutar herramientas en grafos LangGraph
Basado en: Documentación oficial de LangGraph (2025)
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import operator

# Cargar variables de entorno
load_dotenv()


# ============================================================================
# HERRAMIENTAS
# ============================================================================

@tool
def buscar_jurisprudencia(tema: str, corte: str = "todas") -> str:
    """
    Busca jurisprudencia sobre un tema específico.
    
    Args:
        tema: Tema de búsqueda (ej: "tutela", "derecho laboral")
        corte: Corte específica (CC, CE, CSJ, o "todas")
    
    Returns:
        Jurisprudencia encontrada
    """
    
    jurisprudencia = {
        "tutela": [
            "T-123 de 2024 - Protección derechos fundamentales",
            "T-456 de 2024 - Salud - Medicamentos",
            "T-789 de 2023 - Educación - Pensionados"
        ],
        "laboral": [
            "SU-001 de 2024 - Contrato trabajo - Terminación",
            "C-100 de 2023 - Jornada laboral - Máxima"
        ],
        "penal": [
            "SP-234 de 2024 - Habeas Corpus - Libertad",
            "AP-567 de 2023 - Extinción dominio"
        ]
    }
    
    tema_key = tema.lower()
    
    if tema_key in jurisprudencia:
        resultados = jurisprudencia[tema_key]
        return f"Jurisprudencia encontrada ({len(resultados)} fallos):\n\n" + "\n".join([f"• {r}" for r in resultados])
    else:
        return f"No se encontró jurisprudencia sobre '{tema}'. Temas disponibles: {', '.join(jurisprudencia.keys())}"


@tool
def calcular_termino_procesal(dias: int, tipo_proceso: str) -> str:
    """
    Calcula términos procesales según el tipo de proceso.
    
    Args:
        dias: Número de días del término
        tipo_proceso: Tipo de proceso (civil, penal, laboral, administrativo)
    
    Returns:
        Cálculo del término con fecha
    """
    from datetime import datetime, timedelta
    
    # Términos por tipo de proceso (simplificado)
    tipos = {
        "civil": {"factor": 1.0, "descripcion": "Proceso Civil Ordinario"},
        "penal": {"factor": 0.5, "descripcion": "Proceso Penal - Sistema Acusatorio"},
        "laboral": {"factor": 0.75, "descripcion": "Proceso Laboral"},
        "administrativo": {"factor": 1.25, "descripcion": "Proceso Contencioso Administrativo"}
    }
    
    if tipo_proceso.lower() not in tipos:
        return f"Tipo de proceso '{tipo_proceso}' no válido. Opciones: {', '.join(tipos.keys())}"
    
    tipo = tipos[tipo_proceso.lower()]
    dias_efectivos = int(dias * tipo["factor"])
    
    hoy = datetime.now()
    vencimiento = hoy + timedelta(days=dias_efectivos)
    
    return f"""
    ╔═══════════════════════════════════════════════════════════╗
                    CÁLCULO DE TÉRMINO PROCESAL
    ╚═══════════════════════════════════════════════════════════╝
    
    Tipo de Proceso: {tipo['descripcion']}
    Término Original: {dias} días
    Factor Aplicado: {tipo['factor']}
    Término Efectivo: {dias_efectivos} días
    
    Fecha Inicial: {hoy.strftime('%Y-%m-%d')}
    Fecha Vencimiento: {vencimiento.strftime('%Y-%m-%d')}
    
    ⚠️ Nota: Este cálculo no incluye festivos ni vacaciones judiciales.
    """


@tool
def verificar_requisitos(tipo_accion: str) -> str:
    """
    Verifica requisitos para una acción legal específica.
    
    Args:
        tipo_accion: Tipo de acción (tutela, demanda, recurso)
    
    Returns:
        Lista de requisitos
    """
    
    requisitos = {
        "tutela": [
            "✓ Identificación completa del accionante",
            "✓ Relato claro de los hechos",
            "✓ Derechos fundamentales vulnerados",
            "✓ Autoridad responsable",
            "✓ Pruebas documentales (si las hay)",
            "✓ No requiere abogado"
        ],
        "demanda": [
            "✓ Identificación de las partes",
            "✓ Pretensiones claras",
            "✓ Hechos fundamentales",
            "✓ Pruebas",
            "✓ Fundamentos de derecho",
            "✓ Firma de abogado (tarjeta profesional)"
        ],
        "recurso": [
            "✓ Decisión que se recurre",
            "✓ Causal específica de inconformidad",
            "✓ Argumentos de violación",
            "✓ Pruebas nuevas (si aplica)",
            "✓ Cumplimiento de términos"
        ]
    }
    
    if tipo_accion.lower() in requisitos:
        reqs = requisitos[tipo_accion.lower()]
        return f"Requisitos para {tipo_accion.capitalize()}:\n\n" + "\n".join(reqs)
    else:
        return f"Tipo de acción '{tipo_accion}' no reconocido. Opciones: {', '.join(requisitos.keys())}"


# ============================================================================
# GRAFO CON TOOLNODE - VERSIÓN BÁSICA
# ============================================================================

def crear_grafo_basico():
    """
    Crea un grafo LangGraph básico con ToolNode.
    
    Returns:
        Grafo compilado listo para usar
    """
    
    print("\n" + "=" * 80)
    print("CREANDO GRAFO CON TOOLNODE - VERSIÓN BÁSICA")
    print("=" * 80)
    
    # 1. Crear ToolNode con herramientas
    tool_node = ToolNode([buscar_jurisprudencia, calcular_termino_procesal, verificar_requisitos])
    
    # 2. Preparar LLM con herramientas vinculadas
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    ).bind_tools([buscar_jurisprudencia, calcular_termino_procesal, verificar_requisitos])
    
    # 3. Definir nodo del agente
    def agente(state: MessagesState):
        """
        Nodo del agente que decide si llamar herramientas.
        
        Args:
            state: Estado actual con mensajes
        
        Returns:
            Nuevo estado con respuesta del LLM
        """
        print("\n🤖 Agente: Procesando mensaje...")
        response = llm.invoke(state["messages"])
        print(f"   LLM respondió: {type(response).__name__}")
        
        # Verificar si hay tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"   📞 Tool calls detectados: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                print(f"      - {tc.get('name', 'unknown')}")
        
        return {"messages": [response]}
    
    # 4. Función de condición para edges condicionales
    def debe_ejecutar_herramientas(state: MessagesState) -> str:
        """
        Decide si ejecutar herramientas o terminar.
        
        Args:
            state: Estado actual
        
        Returns:
            "herramientas" o "fin"
        """
        ultimo_mensaje = state["messages"][-1]
        
        # Verificar tool calls
        if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
            return "herramientas"
        
        return "fin"
    
    # 5. Construir grafo
    builder = StateGraph(MessagesState)
    
    # Agregar nodos
    builder.add_node("agente", agente)
    builder.add_node("herramientas", tool_node)
    
    # Agregar edges
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
    
    # 6. Compilar
    graph = builder.compile()
    
    print("\n✅ Grafo creado exitosamente")
    print("   Estructura: START → agente → [herramientas ↔ agente] → END")
    
    return graph


# ============================================================================
# GRAFO CON TOOLNODE - VERSIÓN AVANZADA (Con Estado Personalizado)
# ============================================================================

class EstadoLegal(TypedDict):
    """Estado personalizado para el asistente legal."""
    
    messages: Annotated[List, operator.add]
    usuario_id: str
    historial_busquedas: List[str]
    herramientas_usadas: List[str]


def crear_grafo_avanzado():
    """
    Crea un grafo LangGraph avanzado con estado personalizado.
    
    Returns:
        Grafo compilado con estado personalizado
    """
    
    print("\n" + "=" * 80)
    print("CREANDO GRAFO CON TOOLNODE - VERSIÓN AVANZADA")
    print("=" * 80)
    
    # Herramientas que modifican el estado
    @tool
    def registrar_busqueda(tema: str) -> str:
        """Registra una búsqueda en el historial del usuario."""
        return f"Búsqueda registrada: {tema}"
    
    # ToolNode con herramientas adicionales
    tool_node = ToolNode([
        buscar_jurisprudencia,
        calcular_termino_procesal,
        verificar_requisitos,
        registrar_busqueda
    ])
    
    # LLM con herramientas
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    ).bind_tools([
        buscar_jurisprudencia,
        calcular_termino_procesal,
        verificar_requisitos,
        registrar_busqueda
    ])
    
    def agente(state: EstadoLegal):
        """Agente con estado personalizado"""
        print(f"\n🤖 Agente: Usuario {state.get('usuario_id', 'desconocido')}")
        
        response = llm.invoke(state["messages"])
        
        # Registrar herramienta usada
        if hasattr(response, "tool_calls") and response.tool_calls:
            herramientas = state.get("herramientas_usadas", [])
            for tc in response.tool_calls:
                herramientas.append(tc.get('name', 'unknown'))
            
            return {
                "messages": [response],
                "herramientas_usadas": herramientas
            }
        
        return {"messages": [response]}
    
    def debe_ejecutar_herramientas(state: EstadoLegal) -> str:
        ultimo_mensaje = state["messages"][-1]
        
        if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
            return "herramientas"
        
        return "fin"
    
    # Construir grafo con estado personalizado
    builder = StateGraph(EstadoLegal)
    
    builder.add_node("agente", agente)
    builder.add_node("herramientas", tool_node)
    
    builder.add_edge(START, "agente")
    builder.add_conditional_edges(
        "agente",
        debe_ejecutar_herramientas,
        {
            "herramientas": "herramientas",
            "fin": END
        }
    )
    
    builder.add_edge("herramientas", "agente")
    
    graph = builder.compile()
    
    print("\n✅ Grafo avanzado creado exitosamente")
    print("   Estado personalizado: EstadoLegal")
    print("   Trackea: usuario_id, historial_busquedas, herramientas_usadas")
    
    return graph


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def main():
    """Función principal para demostrar ToolNode"""
    
    print("=" * 80)
    print("MÓDULO 5.3 - TOOLNODE EN LANGGRAPH")
    print("=" * 80)
    
    # 1. Grafo básico
    print("\n" + "═" * 80)
    print("PRUEBA 1: GRAFO BÁSICO")
    print("═" * 80)
    
    graph_basico = crear_grafo_basico()
    
    preguntas = [
        "¿Qué es una tutela?",
        "Busca jurisprudencia sobre tutela y salud",
        "Calcula un término de 10 días en proceso laboral",
        "¿Cuáles son los requisitos para una demanda?",
    ]
    
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n{'='*80}")
        print(f"PREGUNTA {i}: {pregunta}")
        print(f"{'='*80}")
        
        response = graph_basico.invoke({
            "messages": [HumanMessage(content=pregunta)]
        })
        
        # Mostrar último mensaje
        ultimo = response["messages"][-1]
        
        if isinstance(ultimo, AIMessage):
            print(f"\n💬 RESPUESTA:\n{ultimo.content[:500]}...")
        elif isinstance(ultimo, ToolMessage):
            print(f"\n🔧 HERRAMIENTA: {ultimo.name}")
            print(f"   Resultado: {ultimo.content[:300]}...")
    
    # 2. Grafo avanzado
    print("\n" + "═" * 80)
    print("PRUEBA 2: GRAFO AVANZADO CON ESTADO")
    print("═" * 80)
    
    graph_avanzado = crear_grafo_avanzado()
    
    # Estado inicial personalizado
    estado_inicial = {
        "messages": [],
        "usuario_id": "usuario_123",
        "historial_busquedas": [],
        "herramientas_usadas": []
    }
    
    # Primera pregunta
    print("\n📝 Primera pregunta:")
    response = graph_avanzado.invoke({
        **estado_inicial,
        "messages": [HumanMessage(content="Busca jurisprudencia laboral")]
    })
    
    print(f"Herramientas usadas: {response.get('herramientas_usadas', [])}")
    
    # Segunda pregunta (con contexto)
    print("\n📝 Segunda pregunta (con contexto):")
    response = graph_avanzado.invoke({
        **response,  # Mantiene estado anterior
        "messages": [HumanMessage(content="Ahora verifica requisitos para demanda")]
    })
    
    print(f"Total herramientas usadas: {len(response.get('herramientas_usadas', []))}")


# ============================================================================
# VENTAJAS DE TOOLNODE
# ============================================================================

"""
✅ VENTAJAS DE USAR TOOLNODE:

1. Ejecución Automática
   - Detecta tool_calls automáticamente
   - Ejecuta herramientas en paralelo
   - Maneja ToolMessage creation

2. Error Handling
   - Captura excepciones de herramientas
   - Retorna errores como ToolMessage
   - Permite retry desde el agente

3. State Injection
   - Inyecta resultados en el estado
   - Mantiene historial de mensajes
   - Soporta estado personalizado

4. Paralelismo
   - Ejecuta múltiples tools concurrentemente
   - Espera todas las respuestas
   - Agrega resultados al estado

5. Integración con LangGraph
   - Compatible con persistence
   - Funciona con interrupts
   - Soporta subgraphs
"""


if __name__ == "__main__":
    main()
