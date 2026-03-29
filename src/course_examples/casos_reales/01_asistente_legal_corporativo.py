"""
Caso Real 1: Asistente Legal Corporativo

Implementación de sistema multi-agente para consultas legales de empleados.
Patrón: Router + Multi-Agent Handoff

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
License: MIT
"""

import os
import logging
from typing import TypedDict, List, Literal, Optional, Annotated
import operator
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langchain_google_genai import ChatGoogleGenerativeAI

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


# =============================================================================
# MODELOS DE DATOS
# =============================================================================

class ConsultaLegalInput(BaseModel):
    """Modelo para consulta legal de empleado."""
    
    pregunta: str = Field(..., description="Pregunta del empleado")
    empleado_id: str = Field(..., description="ID del empleado")
    departamento: str = Field(default="general", description="Departamento del empleado")
    urgencia: Literal["baja", "media", "alta"] = Field(default="media", description="Nivel de urgencia")


class ClasificacionLegal(BaseModel):
    """Modelo para clasificación de consulta legal."""
    
    area: Literal["laboral", "corporativo", "contractual", "compliance", "tributario", "otros"]
    confianza: float = Field(..., ge=0.0, le=1.0, description="Confianza de la clasificación")
    razonamiento: str = Field(..., description="Razón de la clasificación")
    palabras_clave: List[str] = Field(default_factory=list, description="Palabras clave identificadas")


class RespuestaLegal(BaseModel):
    """Modelo para respuesta legal."""
    
    respuesta: str = Field(..., description="Respuesta a la consulta")
    area: str = Field(..., description="Área legal que respondió")
    nivel_confianza: float = Field(..., ge=0.0, le=1.0)
    referencias: List[str] = Field(default_factory=list, description="Referencias legales citadas")
    recomendacion_accion: str = Field(..., description="Acción recomendada si aplica")


# =============================================================================
# ESTADO DEL SISTEMA
# =============================================================================

class AsistenteLegalState(TypedDict):
    """Estado del asistente legal corporativo."""
    
    messages: Annotated[List[BaseMessage], operator.add]
    consulta: str
    empleado_id: str
    departamento: str
    urgencia: str
    clasificacion: Optional[ClasificacionLegal]
    respuesta: Optional[RespuestaLegal]
    agente_asignado: Optional[str]
    metadata: dict


# =============================================================================
# AGENTES ESPECIALIZADOS
# =============================================================================

class AgenteEspecializado:
    """Clase base para agentes especializados."""
    
    def __init__(self, area: str, system_prompt: str):
        self.area = area
        self.system_prompt = system_prompt
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{consulta}")
        ])
        
        self.chain = self.prompt | self.llm.with_structured_output(RespuestaLegal)
    
    def responder(self, consulta: str, contexto: Optional[dict] = None) -> RespuestaLegal:
        """Generar respuesta especializada."""
        
        response = self.chain.invoke({"consulta": consulta})
        response.area = self.area
        
        return response


# Crear agentes especializados
AGENTES = {
    "laboral": AgenteEspecializado(
        area="laboral",
        system_prompt="""Eres un abogado laboralista experto en derecho laboral colombiano.
        
        Tus capacidades:
        - Contratos de trabajo
        - Despidos y liquidaciones
        - Prestaciones sociales (cesantías, primas, vacaciones)
        - Acoso laboral
        - Seguridad social
        
        Responde de forma clara y práctica.
        Cita artículos del Código Sustantivo del Trabajo cuando sea relevante.
        """
    ),
    
    "corporativo": AgenteEspecializado(
        area="corporativo",
        system_prompt="""Eres un abogado corporativo experto en derecho comercial colombiano.
        
        Tus capacidades:
        - Constitución de empresas
        - Estatutos societarios
        - Juntas directivas
        - Compliance corporativo
        - Propiedad intelectual
        
        Responde de forma técnica pero accesible.
        Cita el Código de Comercio cuando sea relevante.
        """
    ),
    
    "contractual": AgenteEspecializado(
        area="contractual",
        system_prompt="""Eres un abogado experto en contratación y obligaciones.
        
        Tus capacidades:
        - Contratos civiles y comerciales
        - Cláusulas contractuales
        - Incumplimiento
        - Garantías
        - Responsabilidad contractual
        
        Responde identificando riesgos y obligaciones.
        Cita el Código Civil cuando sea relevante.
        """
    ),
    
    "compliance": AgenteEspecializado(
        area="compliance",
        system_prompt="""Eres un experto en compliance y regulación.
        
        Tus capacidades:
        - Normativa anticorrupción
        - Protección de datos (Habeas Data)
        - Prevención de lavado de activos
        - Código de ética
        - Denuncias y whistleblowing
        
        Responde identificando obligaciones normativas.
        """
    ),
    
    "tributario": AgenteEspecializado(
        area="tributario",
        system_prompt="""Eres un abogado tributarista experto.
        
        Tus capacidades:
        - Impuestos (renta, IVA, retenciones)
        - Obligaciones formales
        - Sanciones tributarias
        - Beneficios tributarios
        - Calendarios fiscales
        
        Responde de forma precisa y actualizada.
        Cita el Estatuto Tributario cuando sea relevante.
        """
    )
}


# =============================================================================
# NODOS DEL GRAFO
# =============================================================================

class NodosAsistenteLegal:
    """Nodos del grafo del asistente legal."""
    
    def __init__(self):
        self.llm_clasificador = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1
        )
        
        self.prompt_clasificacion = ChatPromptTemplate.from_messages([
            ("system", """Eres un clasificador de consultas legales.
            Analiza la consulta y determina el área legal más apropiada.
            
            Áreas disponibles:
            - laboral: Contratos, despidos, prestaciones, seguridad social
            - corporativo: Empresas, sociedades, juntas directivas
            - contractual: Contratos, obligaciones, cláusulas
            - compliance: Normativa, anticorrupción, protección de datos
            - tributario: Impuestos, obligaciones fiscales
            - otros: Todo lo que no encaje en las anteriores
            
            Responde en formato JSON con área, confianza (0-1), razonamiento y palabras_clave."""),
            ("user", "Consulta: {consulta}")
        ])
        
        self.chain_clasificacion = (
            self.prompt_clasificacion | 
            self.llm_clasificador.with_structured_output(ClasificacionLegal)
        )
    
    def recibir_consulta(self, state: AsistenteLegalState) -> dict:
        """Nodo 1: Recibir y registrar consulta."""
        
        logger.info(f"Recibiendo consulta de empleado {state['empleado_id']}")
        
        return {
            "messages": [
                HumanMessage(content=state['consulta']),
                AIMessage(content="Consulta recibida. Clasificando...")
            ]
        }
    
    def clasificar_consulta(self, state: AsistenteLegalState) -> dict:
        """Nodo 2: Clasificar consulta por área legal."""
        
        logger.info("Clasificando consulta...")
        
        try:
            clasificacion = self.chain_clasificacion.invoke({
                "consulta": state['consulta']
            })
            
            logger.info(f"Clasificación: {clasificacion.area} (confianza: {clasificacion.confianza})")
            
            return {
                "clasificacion": clasificacion,
                "messages": [
                    AIMessage(content=f"Consulta clasificada como: {clasificacion.area}")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            
            # Default a 'otros' si hay error
            return {
                "clasificacion": ClasificacionLegal(
                    area="otros",
                    confianza=1.0,
                    razonamiento=f"Error en clasificación: {str(e)}",
                    palabras_clave=[]
                ),
                "messages": [AIMessage(content="Error clasificando, usando ruta por defecto")]
            }
    
    def asignar_agente(self, state: AsistenteLegalState) -> Command[Literal["ejecutar_consulta", "escalar"]]:
        """Nodo 3: Decidir qué agente asignar."""
        
        logger.info("Decidiendo asignación de agente...")
        
        clasificacion = state['clasificacion']
        
        # Si baja confianza, escalar a humano
        if clasificacion.confianza < 0.5:
            logger.warning("Baja confianza, escalando a humano")
            return Command(goto="escalar")
        
        # Si es 'otros', verificar urgencia
        if clasificacion.area == "otros":
            if state['urgencia'] == "alta":
                logger.warning("Consulta 'otros' con urgencia alta, escalando")
                return Command(goto="escalar")
        
        # Asignar agente especializado
        agente = clasificacion.area
        logger.info(f"Asignando agente: {agente}")
        
        return Command(goto="ejecutar_consulta")
    
    def ejecutar_consulta(self, state: AsistenteLegalState) -> dict:
        """Nodo 4: Ejecutar consulta con agente especializado."""
        
        logger.info("Ejecutando consulta con agente especializado...")
        
        area = state['clasificacion'].area
        agente = AGENTES.get(area, AGENTES["otros"] if "otros" in AGENTES else list(AGENTES.values())[0])
        
        try:
            respuesta = agente.responder(state['consulta'])
            
            logger.info(f"Respuesta generada por agente {area}")
            
            return {
                "respuesta": respuesta,
                "agente_asignado": area,
                "messages": [
                    AIMessage(content=respuesta.respuesta)
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            
            return {
                "respuesta": RespuestaLegal(
                    respuesta=f"Error generando respuesta: {str(e)}",
                    area="error",
                    nivel_confianza=0.0,
                    referencias=[],
                    recomendacion_accion="Contactar abogado humano"
                ),
                "agente_asignado": "error",
                "messages": [AIMessage(content=f"Error: {str(e)}")]
            }
    
    def escalar(self, state: AsistenteLegalState) -> dict:
        """Nodo 5: Escalar a abogado humano."""
        
        logger.warning("Escalando consulta a abogado humano")
        
        # En producción: enviar email, crear ticket, etc.
        respuesta = RespuestaLegal(
            respuesta="Su consulta ha sido escalada a un abogado humano. Recibirá respuesta en 24 horas hábiles.",
            area="escalamiento",
            nivel_confianza=1.0,
            referencias=[],
            recomendacion_accion="Un abogado se contactará con usted"
        )
        
        return {
            "respuesta": respuesta,
            "agente_asignado": "humano",
            "messages": [AIMessage(content=respuesta.respuesta)]
        }
    
    def registrar_metadata(self, state: AsistenteLegalState) -> dict:
        """Nodo 6: Registrar metadata para analytics."""
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "empleado_id": state['empleado_id'],
            "departamento": state['departamento'],
            "urgencia": state['urgencia'],
            "area_legal": state['clasificacion'].area if state['clasificacion'] else "N/A",
            "confianza": state['clasificacion'].confianza if state['clasificacion'] else 0.0,
            "agente": state['agente_asignado'],
            "exitosa": state['respuesta'] is not None and state['respuesta'].nivel_confianza > 0.7 if state['respuesta'] else False
        }
        
        # En producción: guardar en base de datos, enviar a analytics, etc.
        logger.info(f"Metadata registrada: {metadata}")
        
        return {
            "metadata": metadata,
            "messages": state['messages']
        }


# =============================================================================
# SISTEMA PRINCIPAL
# =============================================================================

class AsistenteLegalCorporativo:
    """
    Sistema de Asistente Legal Corporativo multi-agente.
    
    Patrón: Router + Multi-Agent Handoff
    
    Flujo:
    1. Recibir consulta
    2. Clasificar por área legal
    3. Asignar agente especializado
    4. Ejecutar consulta
    5. Registrar metadata
    """
    
    def __init__(self):
        """Inicializar sistema."""
        
        logger.info("Inicializando Asistente Legal Corporativo")
        
        # Inicializar nodos
        self.nodos = NodosAsistenteLegal()
        
        # Construir grafo
        self.graph = self._construir_grafo()
        
        logger.info("Asistente Legal Corporativo inicializado")
    
    def _construir_grafo(self):
        """Construir grafo del sistema."""
        
        builder = StateGraph(AsistenteLegalState)
        
        # Agregar nodos
        builder.add_node("recibir", self.nodos.recibir_consulta)
        builder.add_node("clasificar", self.nodos.clasificar_consulta)
        builder.add_node("ejecutar", self.nodos.ejecutar_consulta)
        builder.add_node("escalar", self.nodos.escalar)
        builder.add_node("registrar", self.nodos.registrar_metadata)
        
        # Agregar edges
        builder.add_edge(START, "recibir")
        builder.add_edge("recibir", "clasificar")
        
        # Edge condicional para asignación
        builder.add_conditional_edges(
            "clasificar",
            self.nodos.asignar_agente,
            {
                "ejecutar_consulta": "ejecutar",
                "escalar": "escalar"
            }
        )
        
        # Ambos caminos van a registro
        builder.add_edge("ejecutar", "registrar")
        builder.add_edge("escalar", "registrar")
        builder.add_edge("registrar", END)
        
        # Compilar con persistencia
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        
        return graph
    
    def consultar(
        self, 
        pregunta: str, 
        empleado_id: str,
        departamento: str = "general",
        urgencia: str = "media",
        thread_id: Optional[str] = None
    ) -> RespuestaLegal:
        """
        Realizar consulta legal.
        
        Args:
            pregunta: Pregunta del empleado
            empleado_id: ID del empleado
            departamento: Departamento del empleado
            urgencia: Nivel de urgencia (baja, media, alta)
            thread_id: ID de hilo para persistencia
        
        Returns:
            Respuesta legal generada
        """
        
        logger.info(f"Consulta recibida: {pregunta[:50]}...")
        
        # Estado inicial
        initial_state = {
            "messages": [],
            "consulta": pregunta,
            "empleado_id": empleado_id,
            "departamento": departamento,
            "urgencia": urgencia,
            "clasificacion": None,
            "respuesta": None,
            "agente_asignado": None,
            "metadata": {}
        }
        
        # Configurar thread
        config = {
            "configurable": {
                "thread_id": thread_id or f"consulta-{empleado_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        }
        
        try:
            # Ejecutar grafo
            result = self.graph.invoke(initial_state, config=config)
            
            logger.info("Consulta completada exitosamente")
            
            return result['respuesta']
            
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            raise


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def main():
    """Ejemplo de uso del asistente legal corporativo."""
    
    print("=" * 80)
    print("CASO REAL 1: ASISTENTE LEGAL CORPORATIVO")
    print("=" * 80)
    
    # Inicializar sistema
    sistema = AsistenteLegalCorporativo()
    
    # Consultas de ejemplo
    consultas = [
        {
            "pregunta": "¿Cuántos días de vacaciones me corresponden si trabajé 1 año?",
            "empleado_id": "EMP001",
            "departamento": "Recursos Humanos",
            "urgencia": "baja"
        },
        {
            "pregunta": "¿Cómo registro una SAS y qué documentos necesito?",
            "empleado_id": "EMP002",
            "departamento": "Jurídico",
            "urgencia": "media"
        },
        {
            "pregunta": "¿Qué hago si un proveedor incumple el contrato?",
            "empleado_id": "EMP003",
            "departamento": "Compras",
            "urgencia": "alta"
        },
        {
            "pregunta": "¿Cuáles son mis obligaciones para reportar operaciones sospechosas?",
            "empleado_id": "EMP004",
            "departamento": "Finanzas",
            "urgencia": "alta"
        }
    ]
    
    # Ejecutar consultas
    for i, consulta in enumerate(consultas, 1):
        print(f"\n{'='*80}")
        print(f"CONSULTA {i}: {consulta['pregunta']}")
        print(f"{'='*80}")
        
        try:
            respuesta = sistema.consultar(**consulta)
            
            print(f"\n✅ Respuesta:")
            print(f"{respuesta.respuesta}")
            
            print(f"\n📊 Metadata:")
            print(f"  Área: {respuesta.area}")
            print(f"  Confianza: {respuesta.nivel_confianza*100:.0f}%")
            
            if respuesta.referencias:
                print(f"  Referencias: {', '.join(respuesta.referencias)}")
            
            print(f"  Recomendación: {respuesta.recomendacion_accion}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
