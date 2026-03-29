"""
Ejercicios Resueltos - Módulos 1-6

Propósito: Reforzar conceptos con ejercicios prácticos completos
Todos los ejercicios son ejecutables y están listos para usar
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# EJERCICIO 1: SISTEMA DE CONSULTAS LEGALES CON MEMORIA
# Combina: Módulos 1 (Prompts/Mensajes) + Módulo 2 (Memoria)
# ============================================================================

class SistemaConsultasLegales:
    """
    Sistema de consultas legales con memoria de conversaciones.
    
    Características:
    - Mantiene contexto de la conversación
    - Especialización por área del derecho
    - Historial de consultas
    """
    
    def __init__(self, especialidad: str = "colombiano"):
        from langchain.memory import ConversationBufferWindowMemory
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, AIMessage
        
        self.especialidad = especialidad
        
        # Memoria (últimos 5 turnos)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
        
        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Eres un asistente legal experto en derecho {especialidad}.
            
            Tus capacidades:
            - Responder consultas legales
            - Citar normas vigentes
            - Explicar procedimientos
            - Orientar sobre mecanismos de protección
            
            Instrucciones:
            - Sé preciso en tus respuestas
            - Cita artículos cuando sea relevante
            - Usa lenguaje técnico pero accesible
            - Si no sabes algo, admítelo honestamente
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        
        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3
        )
        
        # Historial completo
        self.historial_completo = []
    
    def consultar(self, pregunta: str) -> str:
        """
        Realiza una consulta legal.
        
        Args:
            pregunta: Consulta del usuario
        
        Returns:
            Respuesta del asistente
        """
        # Cargar historial
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # Formatear prompt
        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=pregunta
        )
        
        # Obtener respuesta
        response = self.llm.invoke(messages)
        
        # Guardar en memoria
        self.memory.save_context(
            {"input": pregunta},
            {"output": response.content}
        )
        
        # Guardar en historial completo
        self.historial_completo.append({
            "pregunta": pregunta,
            "respuesta": response.content
        })
        
        return response.content
    
    def obtener_historial(self) -> list:
        """Obtiene el historial completo de consultas."""
        return self.historial_completo
    
    def limpiar(self):
        """Limpia la memoria y el historial."""
        self.memory.clear()
        self.historial_completo = []


def demo_sistema_consultas():
    """Demuestra el sistema de consultas legales."""
    
    print("=" * 80)
    print("EJERCICIO 1: SISTEMA DE CONSULTAS LEGALES CON MEMORIA")
    print("=" * 80)
    
    sistema = SistemaConsultasLegales(especialidad="laboral colombiano")
    
    # Consulta 1
    print("\n📋 Consulta 1:")
    respuesta = sistema.consultar("¿Qué es el debido proceso?")
    print(f"R: {respuesta[:200]}...")
    
    # Consulta 2 (con contexto)
    print("\n📋 Consulta 2 (con contexto):")
    respuesta = sistema.consultar("¿En qué artículos está consagrado?")
    print(f"R: {respuesta[:200]}...")
    
    # Consulta 3 (sigue con contexto)
    print("\n📋 Consulta 3 (sigue con contexto):")
    respuesta = sistema.consultar("¿Qué pasa si se vulnera?")
    print(f"R: {respuesta[:200]}...")
    
    # Mostrar historial
    print("\n📜 Historial de consultas:")
    for i, consulta in enumerate(sistema.obtener_historial(), 1):
        print(f"\n{i}. P: {consulta['pregunta']}")
        print(f"   R: {consulta['respuesta'][:100]}...")


# ============================================================================
# EJERCICIO 2: CLASIFICADOR DE DOCUMENTOS CON LANGGRAPH
# Combina: Módulo 4 (LangGraph) + Módulo 1 (LLM)
# ============================================================================

def crear_clasificador_documentos():
    """
    Clasificador de documentos legales con LangGraph.
    
    Clasifica automáticamente documentos en:
    - Contrato
    - Demanda
    - Sentencia
    - Ley/Decreto
    - Otro
    """
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    class DocumentoState(TypedDict):
        texto: str
        tipo: str
        confianza: float
        analisis: str
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    
    def clasificar(state: DocumentoState):
        """Clasifica el tipo de documento."""
        prompt = f"""
        Clasifica este documento legal:
        
        {state['texto'][:300]}...
        
        Tipos posibles: contrato, demanda, sentencia, ley, otro
        
        Responde en formato JSON:
        {{
            "tipo": "tipo_detectado",
            "confianza": 0.0-1.0,
            "razon": "breve_explicacion"
        }}
        """
        
        response = llm.invoke(prompt)
        
        # Parsear respuesta (simplificado)
        import json
        try:
            resultado = json.loads(response.content)
            return {
                "tipo": resultado.get("tipo", "otro"),
                "confianza": resultado.get("confianza", 0.5),
                "analisis": resultado.get("razon", "")
            }
        except:
            return {"tipo": "otro", "confianza": 0.5, "analisis": "Error al clasificar"}
    
    def analizar_contrato(state: DocumentoState):
        return {"analisis": f"📄 CONTRATO: {state['analisis']}"}
    
    def analizar_demanda(state: DocumentoState):
        return {"analisis": f"⚖️ DEMANDA: {state['analisis']}"}
    
    def analizar_sentencia(state: DocumentoState):
        return {"analisis": f"📋 SENTENCIA: {state['analisis']}"}
    
    def analizar_ley(state: DocumentoState):
        return {"analisis": f"📜 LEY/DECRETO: {state['analisis']}"}
    
    def analizar_otro(state: DocumentoState):
        return {"analisis": f"📝 OTRO: {state['analisis']}"}
    
    # Grafo
    builder = StateGraph(DocumentoState)
    
    # Nodos
    builder.add_node("clasificar", clasificar)
    builder.add_node("analizar_contrato", analizar_contrato)
    builder.add_node("analizar_demanda", analizar_demanda)
    builder.add_node("analizar_sentencia", analizar_sentencia)
    builder.add_node("analizar_ley", analizar_ley)
    builder.add_node("analizar_otro", analizar_otro)
    
    # Edges
    builder.add_edge(START, "clasificar")
    
    def router(state: DocumentoState) -> str:
        tipo = state["tipo"].lower()
        confianza = state["confianza"]
        
        # Si baja confianza, va a genérico
        if confianza < 0.6:
            return "analizar_otro"
        
        if "contrato" in tipo:
            return "analizar_contrato"
        elif "demanda" in tipo:
            return "analizar_demanda"
        elif "sentencia" in tipo:
            return "analizar_sentencia"
        elif "ley" in tipo or "decreto" in tipo:
            return "analizar_ley"
        else:
            return "analizar_otro"
    
    builder.add_conditional_edges("clasificar", router, {
        "analizar_contrato": "analizar_contrato",
        "analizar_demanda": "analizar_demanda",
        "analizar_sentencia": "analizar_sentencia",
        "analizar_ley": "analizar_ley",
        "analizar_otro": "analizar_otro"
    })
    
    builder.add_edge("analizar_contrato", END)
    builder.add_edge("analizar_demanda", END)
    builder.add_edge("analizar_sentencia", END)
    builder.add_edge("analizar_ley", END)
    builder.add_edge("analizar_otro", END)
    
    return builder.compile()


def demo_clasificador():
    """Demuestra el clasificador de documentos."""
    
    print("\n" + "=" * 80)
    print("EJERCICIO 2: CLASIFICADOR DE DOCUMENTOS CON LANGGRAPH")
    print("=" * 80)
    
    graph = crear_clasificador_documentos()
    
    documentos = [
        ("CONTRATO DE PRESTACIÓN DE SERVICIOS...", "contrato"),
        ("DEMANDA LABORAL ORDINARIA...", "demanda"),
        ("CORTE CONSTITUCIONAL SENTENCIA T-123...", "sentencia"),
        ("LEY 1564 DE 2012...", "ley"),
        ("Texto legal genérico...", "otro")
    ]
    
    for texto, tipo_esperado in documentos:
        print(f"\n📄 Documento: {texto[:40]}...")
        print(f"   Tipo esperado: {tipo_esperado}")
        
        resultado = graph.invoke({
            "texto": texto,
            "tipo": "",
            "confianza": 0.0,
            "analisis": ""
        })
        
        print(f"   Tipo detectado: {resultado['tipo']}")
        print(f"   Confianza: {resultado['confianza']*100:.0f}%")
        print(f"   Análisis: {resultado['analisis'][:100]}...")


# ============================================================================
# EJERCICIO 3: HERRAMIENTA DE BÚSQUEDA JURISPRUDENCIAL
# Combina: Módulo 5 (Herramientas) + Módulo 1 (LLM)
# ============================================================================

def crear_herramienta_jurisprudencia():
    """
    Herramienta para búsqueda de jurisprudencia.
    
    Características:
    - Búsqueda por tema
    - Filtrado por corte
    - Filtrado por año
    - Resumen de fallos
    """
    from langchain.tools import tool
    from pydantic import BaseModel, Field
    from typing import Literal, List
    
    # Base de datos simulada
    JURISPRUDENCIA = {
        "tutela": [
            {"numero": "T-123/2024", "corte": "CC", "tema": "Salud", "anio": 2024, "resumen": "Protección derecho a la salud"},
            {"numero": "T-456/2024", "corte": "CC", "tema": "Educación", "anio": 2024, "resumen": "Acceso a educación superior"},
            {"numero": "T-789/2023", "corte": "CC", "tema": "Trabajo", "anio": 2023, "resumen": "Protección derechos laborales"}
        ],
        "laboral": [
            {"numero": "SU-001/2024", "corte": "CSJ", "tema": "Despido", "anio": 2024, "resumen": "Despido injustificado"},
            {"numero": "C-100/2023", "corte": "CC", "tema": "Jornada", "anio": 2023, "resumen": "Jornada laboral máxima"}
        ]
    }
    
    class BusquedaInput(BaseModel):
        tema: str = Field(description="Tema de búsqueda (tutela, laboral, etc.)")
        corte: Literal["todas", "CC", "CSJ", "CE"] = Field(default="todas")
        anio_min: int = Field(default=2020, description="Año mínimo")
    
    @tool(args_schema=BusquedaInput)
    def buscar_jurisprudencia(tema: str, corte: str = "todas", anio_min: int = 2020) -> str:
        """
        Busca jurisprudencia por tema, corte y año.
        
        Args:
            tema: Tema de búsqueda
            corte: Corte (CC=Constitucional, CSJ=Casación, CE=Contencioso)
            anio_min: Año mínimo de búsqueda
        """
        resultados = []
        
        if tema.lower() in JURISPRUDENCIA:
            for fallo in JURISPRUDENCIA[tema.lower()]:
                # Aplicar filtros
                if corte != "todas" and fallo["corte"] != corte:
                    continue
                if fallo["anio"] < anio_min:
                    continue
                
                resultados.append(fallo)
        
        if not resultados:
            return f"No se encontró jurisprudencia sobre '{tema}' con los filtros especificados."
        
        # Formatear resultados
        salida = f"JURISPRUDENCIA ENCONTRADA ({len(resultados)} fallos):\n\n"
        
        for fallo in resultados:
            salida += f"┌────────────────────────────────────────────────────┐\n"
            salida += f"│ {fallo['numero']:<50} │\n"
            salida += f"├────────────────────────────────────────────────────┤\n"
            salida += f"│ Corte: {fallo['corte']:<41} │\n"
            salida += f"│ Año: {fallo['anio']:<42} │\n"
            salida += f"│ Tema: {fallo['tema']:<42} │\n"
            salida += f"│ Resumen: {fallo['resumen']:<39} │\n"
            salida += f"└────────────────────────────────────────────────────┘\n\n"
        
        return salida
    
    return buscar_jurisprudencia


def demo_herramienta_jurisprudencia():
    """Demuestra la herramienta de jurisprudencia."""
    
    print("\n" + "=" * 80)
    print("EJERCICIO 3: HERRAMIENTA DE BÚSQUEDA JURISPRUDENCIAL")
    print("=" * 80)
    
    herramienta = crear_herramienta_jurisprudencia()
    
    # Búsqueda 1
    print("\n🔍 Búsqueda 1: Tutelas de 2024")
    resultado = herramienta.invoke({
        "tema": "tutela",
        "corte": "todas",
        "anio_min": 2024
    })
    print(f"Resultados:\n{resultado}")
    
    # Búsqueda 2
    print("\n🔍 Búsqueda 2: Laboral en Corte Suprema")
    resultado = herramienta.invoke({
        "tema": "laboral",
        "corte": "CSJ",
        "anio_min": 2020
    })
    print(f"Resultados:\n{resultado}")


# ============================================================================
# DEMOSTRACIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta todas las demostraciones de ejercicios."""
    
    print("=" * 80)
    print("EJERCICIOS RESUELTOS - MÓDULOS 1-6")
    print("=" * 80)
    print("\nEste archivo contiene ejercicios prácticos que combinan")
    print("múltiples conceptos de los módulos anteriores.\n")
    
    # Ejercicio 1
    demo_sistema_consultas()
    
    # Ejercicio 2
    demo_clasificador()
    
    # Ejercicio 3
    demo_herramienta_jurisprudencia()
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
    print("\n💡 Próximos pasos:")
    print("   1. Modifica los ejercicios para entender el funcionamiento")
    print("   2. Combina ejercicios para crear soluciones más complejas")
    print("   3. Adapta los ejercicios a tus casos de uso específicos")
    print("\n📚 Para más ejercicios, consulta:")
    print("   docs/curso/07-ejercicios-resueltos/README.md")


if __name__ == "__main__":
    main()
