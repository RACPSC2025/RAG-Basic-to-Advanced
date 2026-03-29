"""
Caso Real 2: Clasificador y Enrutador de Documentos Legales

Implementación de sistema para clasificar documentos legales y enrutarlos al proceso adecuado.
Patrón: Parallelization + Routing

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
License: MIT
"""

import os
import logging
from typing import TypedDict, List, Literal, Optional, Annotated
import operator
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('clasificador_documentos.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


# =============================================================================
# MODELOS DE DATOS
# =============================================================================

class ClasificacionDocumento(BaseModel):
    """Modelo para clasificación de documento legal."""
    
    tipo: Literal[
        "tutela", 
        "demanda", 
        "sentencia", 
        "contrato", 
        "ley", 
        "decreto", 
        "derecho_peticion",
        "acto_administrativo",
        "otros"
    ]
    confianza: float = Field(..., ge=0.0, le=1.0)
    razonamiento: str
    palabras_clave: List[str]
    area_legal: Literal["constitucional", "penal", "civil", "laboral", "administrativo", "comercial", "otros"]
    nivel_urgencia: Literal["baja", "media", "alta", "muy_alta"]


class ElementosExtraidos(BaseModel):
    """Modelo para elementos extraídos del documento."""
    
    # Para tutelas
    derechos_vulnerados: Optional[List[str]] = None
    
    # Para demandas
    pretensiones: Optional[List[str]] = None
    hechos: Optional[List[str]] = None
    
    # Para sentencias
    decision: Optional[str] = None
    magistrado_ponente: Optional[str] = None
    numero_sentencia: Optional[str] = None
    
    # Para contratos
    partes: Optional[List[str]] = None
    objeto_contractual: Optional[str] = None
    valor: Optional[str] = None
    plazo: Optional[str] = None
    
    # Para leyes/decretos
    numero_norma: Optional[str] = None
    ano_norma: Optional[str] = None
    tema: Optional[str] = None


class ProcesoAsignado(BaseModel):
    """Modelo para proceso asignado al documento."""
    
    proceso: str
    prioridad: Literal["baja", "media", "alta"]
    tiempo_estimado_procesamiento: str  # En minutos
    responsable: str


# =============================================================================
# ESTADO DEL SISTEMA
# =============================================================================

class ClasificadorState(TypedDict):
    """Estado del clasificador de documentos."""
    
    messages: Annotated[List[BaseMessage], operator.add]
    documento: Document
    texto: str
    clasificacion: Optional[ClasificacionDocumento]
    elementos: Optional[ElementosExtraidos]
    proceso_asignado: Optional[ProcesoAsignado]
    metadata: dict


# =============================================================================
# PROCESOS ESPECIALIZADOS
# =============================================================================

class ProcesosEspecializados:
    """Procesos especializados por tipo de documento."""
    
    @staticmethod
    def procesar_tutela(documento: Document, elementos: ElementosExtraidos) -> str:
        """Procesar acción de tutela."""
        
        informe = "📋 INFORME DE TUTELA\n"
        informe += "=" * 60 + "\n\n"
        
        if elementos.derechos_vulnerados:
            informe += "Derechos Vulnerados:\n"
            for derecho in elementos.derechos_vulnerados:
                informe += f"  • {derecho}\n"
            informe += "\n"
        
        informe += "Acciones Recomendadas:\n"
        informe += "  1. Verificar procedencia de la tutela\n"
        informe += "  2. Identificar autoridad accionada\n"
        informe += "  3. Revisar términos (10 días hábiles)\n"
        informe += "  4. Preparar contestación\n"
        
        return informe
    
    @staticmethod
    def procesar_demanda(documento: Document, elementos: ElementosExtraidos) -> str:
        """Procesar demanda."""
        
        informe = "⚖️ INFORME DE DEMANDA\n"
        informe += "=" * 60 + "\n\n"
        
        if elementos.pretensiones:
            informe += "Pretensiones:\n"
            for pretension in elementos.pretensiones:
                informe += f"  • {pretension}\n"
            informe += "\n"
        
        if elementos.hechos:
            informe += f"Hechos Principales: {len(elementos.hechos)} identificados\n\n"
        
        informe += "Acciones Recomendadas:\n"
        informe += "  1. Analizar viabilidad de defensa\n"
        informe += "  2. Verificar términos de contestación\n"
        informe += "  3. Identificar excepciones de mérito\n"
        informe += "  4. Recopilar pruebas\n"
        
        return informe
    
    @staticmethod
    def procesar_sentencia(documento: Document, elementos: ElementosExtraidos) -> str:
        """Procesar sentencia."""
        
        informe = "📜 INFORME DE SENTENCIA\n"
        informe += "=" * 60 + "\n\n"
        
        if elementos.numero_sentencia:
            informe += f"Número: {elementos.numero_sentencia}\n"
        
        if elementos.magistrado_ponente:
            informe += f"Magistrado Ponente: {elementos.magistrado_ponente}\n"
        
        if elementos.decision:
            informe += f"\nDecisión:\n{elementos.decision}\n\n"
        
        informe += "Acciones Recomendadas:\n"
        informe += "  1. Analizar cosa juzgada\n"
        informe += "  2. Verificar recursos disponibles\n"
        informe += "  3. Evaluar cumplimiento\n"
        
        return informe
    
    @staticmethod
    def procesar_contrato(documento: Document, elementos: ElementosExtraidos) -> str:
        """Procesar contrato."""
        
        informe = "📄 INFORME DE CONTRATO\n"
        informe += "=" * 60 + "\n\n"
        
        if elementos.partes:
            informe += f"Partes: {len(elementos.partes)} identificadas\n"
        
        if elementos.objeto_contractual:
            informe += f"Objeto: {elementos.objeto_contractual}\n"
        
        if elementos.valor:
            informe += f"Valor: {elementos.valor}\n"
        
        if elementos.plazo:
            informe += f"Plazo: {elementos.plazo}\n"
        
        informe += "\nAcciones Recomendadas:\n"
        informe += "  1. Verificar cumplimiento de obligaciones\n"
        informe += "  2. Identificar fechas de vencimiento\n"
        informe += "  3. Revisar cláusulas de terminación\n"
        informe += "  4. Evaluar riesgos contractuales\n"
        
        return informe
    
    @staticmethod
    def procesar_norma(documento: Document, elementos: ElementosExtraidos, tipo: str) -> str:
        """Procesar ley o decreto."""
        
        informe = f"📜 INFORME DE {tipo.upper()}\n"
        informe += "=" * 60 + "\n\n"
        
        if elementos.numero_norma and elementos.ano_norma:
            informe += f"{tipo} {elementos.numero_norma} de {elementos.ano_norma}\n"
        
        if elementos.tema:
            informe += f"Tema: {elementos.tema}\n\n"
        
        informe += "Acciones Recomendadas:\n"
        informe += "  1. Identificar artículos aplicables al caso\n"
        informe += "  2. Verificar vigencia\n"
        informe += "  3. Buscar jurisprudencia relacionada\n"
        
        return informe


# =============================================================================
# NODOS DEL SISTEMA
# =============================================================================

class NodosClasificador:
    """Nodos del sistema clasificador."""
    
    def __init__(self):
        """Inicializar nodos."""
        
        self.llm_clasificador = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1
        )
        
        self.llm_extractor = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3
        )
        
        self.prompt_clasificacion = ChatPromptTemplate.from_messages([
            ("system", """Eres un clasificador experto de documentos legales colombianos.
            Analiza el documento y determina su tipo, área legal, y nivel de urgencia.
            
            Tipos disponibles:
            - tutela: Acciones de tutela (derechos fundamentales)
            - demanda: Demandas de cualquier tipo
            - sentencia: Fallos de cortes y tribunales
            - contrato: Contratos civiles, comerciales, laborales
            - ley: Leyes del Congreso
            - decreto: Decretos del Gobierno
            - derecho_peticion: Derechos de petición
            - acto_administrativo: Actos de autoridades
            - otros: Documentos que no encajen
            
            Áreas legales:
            - constitucional: Derecho constitucional
            - penal: Derecho penal
            - civil: Derecho civil
            - laboral: Derecho laboral
            - administrativo: Derecho administrativo
            - comercial: Derecho comercial
            - otros: Otras áreas
            
            Responde en formato JSON."""),
            ("user", "Documento:\n{texto}")
        ])
        
        self.chain_clasificacion = (
            self.prompt_clasificacion | 
            self.llm_clasificador.with_structured_output(ClasificacionDocumento)
        )
        
        self.prompt_extraccion = ChatPromptTemplate.from_messages([
            ("system", """Extrae los elementos clave del documento legal.
            Responde en formato JSON con los campos que apliquen según el tipo de documento.
            
            Para tutelas: derechos_vulnerados
            Para demandas: pretensiones, hechos
            Para sentencias: decision, magistrado_ponente, numero_sentencia
            Para contratos: partes, objeto_contractual, valor, plazo
            Para leyes/decretos: numero_norma, ano_norma, tema"""),
            ("user", "Documento:\n{texto}\n\nTipo: {tipo}")
        ])
        
        self.chain_extraccion = (
            self.prompt_extraccion | 
            self.llm_extractor.with_structured_output(ElementosExtraidos)
        )
    
    def cargar_documento(self, state: ClasificadorState) -> dict:
        """Nodo 1: Cargar y pre-procesar documento."""
        
        logger.info("Cargando documento...")
        
        documento = state['documento']
        texto = documento.page_content
        
        # Pre-procesamiento
        texto_limpio = " ".join(texto.split())  # Eliminar espacios múltiples
        
        logger.info(f"Documento cargado: {len(texto_limpio)} caracteres")
        
        return {
            "texto": texto_limpio,
            "messages": [
                AIMessage(content=f"Documento cargado ({len(texto_limpio)} caracteres)")
            ]
        }
    
    def clasificar_documento(self, state: ClasificadorState) -> dict:
        """Nodo 2: Clasificar documento."""
        
        logger.info("Clasificando documento...")
        
        try:
            clasificacion = self.chain_clasificacion.invoke({
                "texto": state['texto'][:5000]  # Limitar longitud
            })
            
            logger.info(f"Clasificación: {clasificacion.tipo} ({clasificacion.area_legal})")
            
            return {
                "clasificacion": clasificacion,
                "messages": [
                    AIMessage(content=f"Documento clasificado como: {clasificacion.tipo}")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            
            return {
                "clasificacion": ClasificacionDocumento(
                    tipo="otros",
                    confianza=1.0,
                    razonamiento=f"Error en clasificación: {str(e)}",
                    palabras_clave=[],
                    area_legal="otros",
                    nivel_urgencia="media"
                ),
                "messages": [AIMessage(content="Error clasificando, usando ruta por defecto")]
            }
    
    def extraer_elementos(self, state: ClasificadorState) -> dict:
        """Nodo 3: Extraer elementos clave del documento."""
        
        logger.info("Extrayendo elementos clave...")
        
        try:
            elementos = self.chain_extraccion.invoke({
                "texto": state['texto'][:5000],
                "tipo": state['clasificacion'].tipo
            })
            
            logger.info("Elementos extraídos exitosamente")
            
            return {
                "elementos": elementos,
                "messages": [AIMessage(content="Elementos clave extraídos")]
            }
            
        except Exception as e:
            logger.error(f"Error en extracción: {e}")
            
            return {
                "elementos": ElementosExtraidos(),
                "messages": [AIMessage(content=f"Error extrayendo: {str(e)}")]
            }
    
    def asignar_proceso(self, state: ClasificadorState) -> dict:
        """Nodo 4: Asignar proceso especializado."""
        
        logger.info("Asignando proceso especializado...")
        
        clasificacion = state['clasificacion']
        elementos = state['elementos']
        
        # Determinar proceso según tipo
        procesos = {
            "tutela": {
                "proceso": "Proceso de Tutela",
                "prioridad": "alta",
                "tiempo": "30",
                "responsable": "Abogado Constitucionalista"
            },
            "demanda": {
                "proceso": "Proceso Contencioso",
                "prioridad": "alta",
                "tiempo": "60",
                "responsable": "Abogado Litigante"
            },
            "sentencia": {
                "proceso": "Análisis de Fallo",
                "prioridad": "media",
                "tiempo": "45",
                "responsable": "Abogado Analista"
            },
            "contrato": {
                "proceso": "Revisión Contractual",
                "prioridad": "media",
                "tiempo": "40",
                "responsable": "Abogado Contractual"
            },
            "ley": {
                "proceso": "Investigación Normativa",
                "prioridad": "baja",
                "tiempo": "20",
                "responsable": "Paralegal"
            },
            "decreto": {
                "proceso": "Investigación Normativa",
                "prioridad": "baja",
                "tiempo": "20",
                "responsable": "Paralegal"
            },
            "derecho_peticion": {
                "proceso": "Respuesta a Petición",
                "prioridad": "alta",
                "tiempo": "15",
                "responsable": "Abogado Administrativo"
            }
        }
        
        # Obtener proceso o default
        proceso_info = procesos.get(clasificacion.tipo, {
            "proceso": "Proceso General",
            "prioridad": "media",
            "tiempo": "30",
            "responsable": "Abogado General"
        })
        
        proceso = ProcesoAsignado(**proceso_info)
        
        logger.info(f"Proceso asignado: {proceso.proceso}")
        
        return {
            "proceso_asignado": proceso,
            "messages": [AIMessage(content=f"Proceso asignado: {proceso.proceso}")]
        }
    
    def ejecutar_proceso(self, state: ClasificadorState) -> dict:
        """Nodo 5: Ejecutar proceso especializado."""
        
        logger.info("Ejecutando proceso especializado...")
        
        clasificacion = state['clasificacion']
        elementos = state['elementos']
        documento = state['documento']
        
        # Ejecutar proceso según tipo
        procesos = ProcesosEspecializados()
        
        if clasificacion.tipo == "tutela":
            informe = procesos.procesar_tutela(documento, elementos)
        elif clasificacion.tipo == "demanda":
            informe = procesos.procesar_demanda(documento, elementos)
        elif clasificacion.tipo == "sentencia":
            informe = procesos.procesar_sentencia(documento, elementos)
        elif clasificacion.tipo == "contrato":
            informe = procesos.procesar_contrato(documento, elementos)
        elif clasificacion.tipo in ["ley", "decreto"]:
            informe = procesos.procesar_norma(documento, elementos, clasificacion.tipo)
        else:
            informe = "Documento clasificado como 'otros'. Requiere revisión manual."
        
        logger.info("Proceso ejecutado exitosamente")
        
        return {
            "messages": [
                AIMessage(content=f"Proceso ejecutado: {state['proceso_asignado'].proceso}")
            ],
            "metadata": {
                "informe": informe,
                "completado": True
            }
        }
    
    def registrar_documento(self, state: ClasificadorState) -> dict:
        """Nodo 6: Registrar documento en sistema."""
        
        logger.info("Registrando documento...")
        
        metadata = {
            "fecha_procesamiento": datetime.now().isoformat(),
            "tipo_documento": state['clasificacion'].tipo if state['clasificacion'] else "N/A",
            "area_legal": state['clasificacion'].area_legal if state['clasificacion'] else "N/A",
            "urgencia": state['clasificacion'].nivel_urgencia if state['clasificacion'] else "N/A",
            "proceso_asignado": state['proceso_asignado'].proceso if state['proceso_asignado'] else "N/A",
            "responsable": state['proceso_asignado'].responsable if state['proceso_asignado'] else "N/A",
            "tiempo_estimado": state['proceso_asignado'].tiempo_estimado_procesamiento if state['proceso_asignado'] else "N/A"
        }
        
        # En producción: guardar en base de datos, sistema de gestión documental, etc.
        logger.info(f"Documento registrado: {metadata}")
        
        return {
            "metadata": {**state.get('metadata', {}), **metadata}
        }


# =============================================================================
# SISTEMA PRINCIPAL
# =============================================================================

class ClasificadorDocumentosLegales:
    """
    Sistema de Clasificación y Enrutamiento de Documentos Legales.
    
    Patrón: Parallelization + Routing
    
    Flujo:
    1. Cargar documento
    2. Clasificar tipo y área
    3. Extraer elementos clave
    4. Asignar proceso especializado
    5. Ejecutar proceso
    6. Registrar documento
    """
    
    def __init__(self):
        """Inicializar sistema."""
        
        logger.info("Inicializando Clasificador de Documentos Legales")
        
        # Inicializar nodos
        self.nodos = NodosClasificador()
        
        # Construir grafo
        self.graph = self._construir_grafo()
        
        logger.info("Clasificador inicializado")
    
    def _construir_grafo(self):
        """Construir grafo del sistema."""
        
        builder = StateGraph(ClasificadorState)
        
        # Agregar nodos
        builder.add_node("cargar", self.nodos.cargar_documento)
        builder.add_node("clasificar", self.nodos.clasificar_documento)
        builder.add_node("extraer", self.nodos.extraer_elementos)
        builder.add_node("asignar", self.nodos.asignar_proceso)
        builder.add_node("ejecutar", self.nodos.ejecutar_proceso)
        builder.add_node("registrar", self.nodos.registrar_documento)
        
        # Agregar edges
        builder.add_edge(START, "cargar")
        builder.add_edge("cargar", "clasificar")
        builder.add_edge("clasificar", "extraer")
        builder.add_edge("extraer", "asignar")
        builder.add_edge("asignar", "ejecutar")
        builder.add_edge("ejecutar", "registrar")
        builder.add_edge("registrar", END)
        
        # Compilar
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        
        return graph
    
    def procesar_documento(
        self, 
        texto: str,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None
    ) -> dict:
        """
        Procesar documento legal.
        
        Args:
            texto: Contenido del documento
            metadata: Metadatos del documento (fuente, fecha, etc.)
            thread_id: ID de hilo para persistencia
        
        Returns:
            Resultado del procesamiento
        """
        
        logger.info(f"Procesando documento ({len(texto)} caracteres)...")
        
        # Crear documento
        documento = Document(
            page_content=texto,
            metadata=metadata or {}
        )
        
        # Estado inicial
        initial_state = {
            "messages": [],
            "documento": documento,
            "texto": "",
            "clasificacion": None,
            "elementos": None,
            "proceso_asignado": None,
            "metadata": {}
        }
        
        # Configurar thread
        config = {
            "configurable": {
                "thread_id": thread_id or f"doc-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        }
        
        try:
            # Ejecutar grafo
            result = self.graph.invoke(initial_state, config=config)
            
            logger.info("Documento procesado exitosamente")
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            raise


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def main():
    """Ejemplo de uso del clasificador."""
    
    print("=" * 80)
    print("CASO REAL 2: CLASIFICADOR Y ENRUTADOR DE DOCUMENTOS LEGALES")
    print("=" * 80)
    
    # Inicializar sistema
    sistema = ClasificadorDocumentosLegales()
    
    # Documentos de ejemplo
    documentos = [
        {
            "texto": """
            ACCIÓN DE TUTELA
            
            Yo, JUAN PÉREZ, mayor de edad, identificado con cédula de ciudadanía No. 123.456.789,
            por medio del presente escrito interpongo ACCIÓN DE TUTELA contra la EPS SALUD TOTAL,
            por la vulneración de mis derechos fundamentales a la SALUD y a la VIDA.
            
            HECHOS:
            1. Padezco de una enfermedad que requiere tratamiento especializado.
            2. La EPS ha negado el medicamento prescrito por mi médico tratante.
            3. He agotado los recursos de reposición sin obtener respuesta favorable.
            
            DERECHOS VULNERADOS:
            - Derecho a la salud (Artículo 49 Constitución)
            - Derecho a la vida (Artículo 11 Constitución)
            
            PRETENSIONES:
            1. Ordenar a la EPS entregar el medicamento.
            2. Ordenar la continuidad del tratamiento.
            """,
            "metadata": {"fuente": "cliente_001", "fecha": "2024-01-15"}
        },
        {
            "texto": """
            CONTRATO DE PRESTACIÓN DE SERVICIOS
            
            En Bogotá, D.C., a 10 de enero de 2024, entre los suscritos:
            
            POR UNA PARTE: EMPRESA COLOMBIANA S.A.S., sociedad comercial identificada con NIT 900.123.456-7,
            representada por su gerente CARLOS RODRÍGUEZ.
            
            POR LA OTRA: CONSULTORES ASOCIADOS LTDA., sociedad comercial identificada con NIT 800.987.654-3,
            representada por su gerente MARÍA GÓMEZ.
            
            OBJETO: El presente contrato tiene por objeto la prestación de servicios de consultoría jurídica.
            
            VALOR: El valor del contrato es de CINCUENTA MILLONES DE PESOS ($50.000.000).
            
            PLAZO: El contrato tendrá una duración de 6 meses, iniciando el 15 de enero de 2024.
            """,
            "metadata": {"fuente": "departamento_juridico", "fecha": "2024-01-10"}
        },
        {
            "texto": """
            CORTE CONSTITUCIONAL
            SENTENCIA T-456 DE 2024
            
            Magistrado Ponente: Dr. JUAN CARLOS HENAO PÉREZ
            
            Bogotá, D.C., 20 de febrero de 2024
            
            La Sala Séptima de Revisión de la Corte Constitucional ha pronunciado la siguiente:
            
            SENTENCIA
            
            En el proceso de revisión de la sentencia de tutela proferida por el Juzgado Primero Penal del Circuito de Bogotá.
            
            DECISIÓN:
            1. CONFIRMAR la sentencia de primera instancia.
            2. ORDENAR a la EPS demandada entregar el medicamento en 48 horas.
            3. TUTELAR los derechos fundamentales a la salud y vida del accionante.
            """,
            "metadata": {"fuente": "corte_constitucional", "fecha": "2024-02-20"}
        }
    ]
    
    # Procesar documentos
    for i, doc in enumerate(documentos, 1):
        print(f"\n{'='*80}")
        print(f"DOCUMENTO {i}: {doc['metadata']['fuente']}")
        print(f"{'='*80}")
        
        try:
            resultado = sistema.procesar_documento(
                texto=doc['texto'],
                metadata=doc['metadata']
            )
            
            print(f"\n✅ Clasificación:")
            print(f"  Tipo: {resultado['clasificacion'].tipo}")
            print(f"  Área: {resultado['clasificacion'].area_legal}")
            print(f"  Urgencia: {resultado['clasificacion'].nivel_urgencia}")
            print(f"  Confianza: {resultado['clasificacion'].confianza*100:.0f}%")
            
            print(f"\n📋 Proceso Asignado:")
            print(f"  Proceso: {resultado['proceso_asignado'].proceso}")
            print(f"  Prioridad: {resultado['proceso_asignado'].prioridad}")
            print(f"  Tiempo estimado: {resultado['proceso_asignado'].tiempo_estimado_procesamiento} minutos")
            print(f"  Responsable: {resultado['proceso_asignado'].responsable}")
            
            print(f"\n📄 Informe:")
            print(f"  {resultado['metadata']['informe'][:500]}...")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
