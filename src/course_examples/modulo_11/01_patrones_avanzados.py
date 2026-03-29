"""
Módulo 11 - Patrones Avanzados de Agentes

Implementación de patrones avanzados de agentes con LangGraph.
Código de producción listo para deploy.

Basado en: https://docs.langchain.com/oss/python/langgraph/agentic-rag

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
License: MIT
"""

import os
import logging
from typing import TypedDict, List, Literal, Annotated, Optional
from dataclasses import dataclass, field
import operator
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class AgentConfig:
    """Configuración del agente de producción."""
    
    # Modelo LLM
    llm_model: str = "gemini-2.0-flash-exp"
    llm_temperature: float = 0.3
    llm_max_tokens: Optional[int] = None
    
    # Embeddings
    embedding_model: str = "models/embedding-001"
    
    # Retrieval
    retrieval_k: int = 3  # Top-K documentos
    max_retries: int = 3  # Reintentos máximos
    
    # Thresholds
    relevance_threshold: float = 0.7  # Threshold para relevancia
    
    # Control de flujo
    max_iterations: int = 5  # Máximas iteraciones de refinamiento
    
    # Paths
    qdrant_path: str = "./qdrant_storage"
    collection_name: str = "documentos_legales"
    
    # Logging
    log_level: str = "INFO"
    enable_langsmith: bool = False


# =============================================================================
# MODELOS DE DATOS
# =============================================================================

class DocumentGrade(BaseModel):
    """Modelo para evaluar relevancia de documentos."""
    
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score de relevancia entre 0 y 1"
    )
    binary: Literal["yes", "no"] = Field(
        ...,
        description="Decisión binaria: yes si es relevante, no si no"
    )
    reason: str = Field(
        ...,
        description="Razón de la evaluación en 1-2 frases"
    )


class QueryTransformation(BaseModel):
    """Modelo para transformar queries."""
    
    rewritten_query: str = Field(
        ...,
        description="Query reescrita optimizada para retrieval"
    )
    reasoning: str = Field(
        ...,
        description="Razón de la transformación"
    )


class AnswerQuality(BaseModel):
    """Modelo para evaluar calidad de respuesta."""
    
    precision: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    clarity: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    feedback: str = Field(..., description="Feedback constructivo")


# =============================================================================
# ESTADO DEL AGENTE
# =============================================================================

class AgentState(TypedDict):
    """
    Estado completo del agente.
    
    Atributos:
        messages: Historial de mensajes de la conversación
        query: Query original del usuario
        current_query: Query actual (puede ser transformada)
        documents: Documentos recuperados
        graded_docs: Documentos evaluados con scores
        answer: Respuesta generada
        iteration: Número de iteración actual
        max_iterations: Máximo de iteraciones permitidas
        metadata: Metadatos adicionales
    """
    
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    current_query: str
    documents: List[Document]
    graded_docs: List[tuple[Document, DocumentGrade]]
    answer: Optional[str]
    iteration: int
    max_iterations: int
    metadata: dict


# =============================================================================
# COMPONENTES DEL AGENTE
# =============================================================================

class RAGAgentComponents:
    """
    Componentes reutilizables para el agente RAG.
    
    Esta clase encapsula todos los componentes necesarios para el agente:
    - LLM
    - Embeddings
    - Vector Store
    - Prompts
    - Chains
    """
    
    def __init__(self, config: AgentConfig):
        """
        Inicializar componentes del agente.
        
        Args:
            config: Configuración del agente
        """
        self.config = config
        
        # Inicializar LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Inicializar embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model,
            task_type="retrieval_document",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Inicializar vector store
        self.vector_store = self._initialize_vector_store()
        
        # Crear prompts
        self.prompts = self._create_prompts()
        
        # Crear chains
        self.chains = self._create_chains()
        
        logger.info("Componentes del agente inicializados correctamente")
    
    def _initialize_vector_store(self) -> QdrantVectorStore:
        """
        Inicializar vector store con Qdrant.
        
        Returns:
            Vector store configurado
        """
        try:
            # Crear directorio si no existe
            Path(self.config.qdrant_path).mkdir(parents=True, exist_ok=True)
            
            # Cliente local con persistencia
            client = QdrantClient(
                path=self.config.qdrant_path,
                port=6333
            )
            
            # Crear o cargar colección
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings
            )
            
            logger.info(f"Vector store inicializado: {self.config.collection_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error inicializando vector store: {e}")
            raise
    
    def _create_prompts(self) -> dict:
        """
        Crear todos los prompts del agente.
        
        Returns:
            Diccionario con prompts
        """
        prompts = {
            "query_generation": ChatPromptTemplate.from_messages([
                ("system", """Eres un experto en recuperación de información legal.
                Tu tarea es generar una query optimizada para buscar en una base de datos legal.
                
                Instrucciones:
                - Usa términos legales precisos
                - Incluye conceptos clave
                - Elimina información irrelevante
                - Mantén la query concisa pero completa"""),
                ("user", "Query original: {query}\n\nQuery optimizada:")
            ]),
            
            "document_grading": ChatPromptTemplate.from_messages([
                ("system", """Eres un evaluador de documentos legales.
                Evalúa la relevancia de cada documento para la query dada.
                
                Criterios:
                - El documento debe contener información relevante para la query
                - Debe ser de fuentes confiables (constitución, leyes, jurisprudencia)
                - Debe ser actual (preferiblemente posterior a 1991)
                
                Responde en formato JSON con score (0-1), binary (yes/no), y reason."""),
                ("user", "Query: {query}\n\nDocumento: {document}\n\nEvaluación:")
            ]),
            
            "answer_generation": ChatPromptTemplate.from_messages([
                ("system", """Eres un asistente legal experto en derecho colombiano.
                Genera una respuesta completa y precisa basada ÚNICAMENTE en el contexto proporcionado.
                
                Instrucciones:
                - Cita las fuentes cuando sea posible
                - Usa lenguaje técnico pero accesible
                - Si la información no está en el contexto, indícalo claramente
                - Sé preciso y conciso"""),
                ("user", "Contexto: {context}\n\nQuery: {query}\n\nRespuesta:")
            ]),
            
            "query_transformation": ChatPromptTemplate.from_messages([
                ("system", """Transforma la query para mejorar el retrieval.
                Explica tu razonamiento."""),
                ("user", "Query original: {query}\n\nQuery transformada:")
            ]),
            
            "answer_evaluation": ChatPromptTemplate.from_messages([
                ("system", """Evalúa la calidad de la respuesta generada.
                Considera precisión, completitud y claridad.
                
                Responde en formato JSON con scores (0-1) y feedback."""),
                ("user", "Query: {query}\n\nRespuesta: {answer}\n\nEvaluación:")
            ])
        }
        
        logger.info(f"Prompts creados: {len(prompts)}")
        return prompts
    
    def _create_chains(self) -> dict:
        """
        Crear todas las chains del agente.
        
        Returns:
            Diccionario con chains
        """
        chains = {
            "query_generation": (
                self.prompts["query_generation"] | 
                self.llm.with_structured_output(QueryTransformation)
            ),
            
            "document_grading": (
                self.prompts["document_grading"] | 
                self.llm.with_structured_output(DocumentGrade)
            ),
            
            "answer_generation": (
                self.prompts["answer_generation"] | 
                self.llm
            ),
            
            "query_transformation": (
                self.prompts["query_transformation"] | 
                self.llm.with_structured_output(QueryTransformation)
            ),
            
            "answer_evaluation": (
                self.prompts["answer_evaluation"] | 
                self.llm.with_structured_output(AnswerQuality)
            )
        }
        
        logger.info(f"Chains creadas: {len(chains)}")
        return chains
    
    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Recuperar documentos del vector store.
        
        Args:
            query: Query de búsqueda
            k: Número de documentos (default: config.retrieval_k)
        
        Returns:
            Lista de documentos recuperados
        """
        k = k or self.config.retrieval_k
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Documentos recuperados: {len(docs)}")
            return docs
            
        except Exception as e:
            logger.error(f"Error en retrieval: {e}")
            return []
    
    def grade_document(
        self, 
        document: Document, 
        query: str
    ) -> DocumentGrade:
        """
        Evaluar relevancia de un documento.
        
        Args:
            document: Documento a evaluar
            query: Query original
        
        Returns:
            Evaluación del documento
        """
        try:
            chain = self.chains["document_grading"]
            grade = chain.invoke({
                "query": query,
                "document": document.page_content[:2000]  # Limitar longitud
            })
            
            logger.debug(f"Documento evaluado: score={grade.score}, binary={grade.binary}")
            return grade
            
        except Exception as e:
            logger.error(f"Error evaluando documento: {e}")
            # Default: no relevante si hay error
            return DocumentGrade(
                score=0.0,
                binary="no",
                reason=f"Error en evaluación: {str(e)}"
            )


# =============================================================================
# NODOS DEL AGENTE
# =============================================================================

class AgentNodes:
    """
    Nodos del grafo del agente.
    
    Cada nodo es una función pura que:
    1. Recibe el estado actual
    2. Procesa información
    3. Retorna actualización del estado
    """
    
    def __init__(self, components: RAGAgentComponents):
        """
        Inicializar nodos con componentes.
        
        Args:
            components: Componentes del agente
        """
        self.components = components
        logger.info("Nodos del agente inicializados")
    
    def generate_query(self, state: AgentState) -> dict:
        """
        Nodo 1: Generar query optimizada.
        
        Args:
            state: Estado actual
        
        Returns:
            Actualización del estado
        """
        logger.info("Nodo: generate_query")
        
        try:
            chain = self.components.chains["query_generation"]
            result = chain.invoke({"query": state["query"]})
            
            logger.info(f"Query generada: {result.rewritten_query}")
            
            return {
                "current_query": result.rewritten_query,
                "messages": [
                    AIMessage(content=f"Query optimizada: {result.rewritten_query}")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error en generate_query: {e}")
            return {
                "current_query": state["query"],
                "messages": [AIMessage(content=f"Error generando query: {str(e)}")]
            }
    
    def retrieve_documents(self, state: AgentState) -> dict:
        """
        Nodo 2: Recuperar documentos.
        
        Args:
            state: Estado actual
        
        Returns:
            Actualización del estado
        """
        logger.info("Nodo: retrieve_documents")
        
        docs = self.components.retrieve_documents(state["current_query"])
        
        if not docs:
            logger.warning("No se encontraron documentos")
            return {
                "documents": [],
                "messages": [AIMessage(content="No se encontraron documentos relevantes")]
            }
        
        logger.info(f"Documentos recuperados: {len(docs)}")
        
        return {
            "documents": docs,
            "messages": [AIMessage(content=f"Recuperados {len(docs)} documentos")]
        }
    
    def grade_documents(self, state: AgentState) -> dict:
        """
        Nodo 3: Evaluar documentos recuperados.
        
        Args:
            state: Estado actual
        
        Returns:
            Actualización del estado
        """
        logger.info("Nodo: grade_documents")
        
        graded_docs = []
        relevant_count = 0
        
        for doc in state["documents"]:
            grade = self.components.grade_document(doc, state["query"])
            graded_docs.append((doc, grade))
            
            if grade.binary == "yes":
                relevant_count += 1
        
        logger.info(f"Documentos relevantes: {relevant_count}/{len(graded_docs)}")
        
        return {
            "graded_docs": graded_docs,
            "messages": [
                AIMessage(
                    content=f"Evaluados {len(graded_docs)} documentos, "
                           f"{relevant_count} relevantes"
                )
            ]
        }
    
    def check_documents(self, state: AgentState) -> Command[Literal["generate_answer", "transform_query"]]:
        """
        Nodo 4: Verificar si hay documentos relevantes.
        
        Args:
            state: Estado actual
        
        Returns:
            Comando para siguiente nodo
        """
        logger.info("Nodo: check_documents")
        
        # Contar documentos relevantes
        relevant_count = sum(
            1 for _, grade in state["graded_docs"] 
            if grade.binary == "yes"
        )
        
        # Verificar si hay suficientes documentos relevantes
        if relevant_count >= 1:
            logger.info("Documentos suficientes, generando respuesta")
            return Command(goto="generate_answer")
        
        # Verificar si alcanzó máximo de iteraciones
        if state["iteration"] >= state["max_iterations"]:
            logger.warning("Máximo de iteraciones alcanzado")
            return Command(goto="generate_answer")
        
        # Transformar query y reintentar
        logger.info("Transformando query para reintentar")
        return Command(goto="transform_query")
    
    def transform_query(self, state: AgentState) -> dict:
        """
        Nodo 5: Transformar query para mejorar retrieval.
        
        Args:
            state: Estado actual
        
        Returns:
            Actualización del estado
        """
        logger.info("Nodo: transform_query")
        
        try:
            chain = self.components.chains["query_transformation"]
            result = chain.invoke({"query": state["current_query"]})
            
            logger.info(f"Query transformada: {result.rewritten_query}")
            
            return {
                "current_query": result.rewritten_query,
                "iteration": state["iteration"] + 1,
                "messages": [
                    AIMessage(content=f"Query transformada: {result.rewritten_query}")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error en transform_query: {e}")
            return {
                "iteration": state["iteration"] + 1,
                "messages": [AIMessage(content=f"Error transformando query: {str(e)}")]
            }
    
    def generate_answer(self, state: AgentState) -> dict:
        """
        Nodo 6: Generar respuesta final.
        
        Args:
            state: Estado actual
        
        Returns:
            Actualización del estado
        """
        logger.info("Nodo: generate_answer")
        
        try:
            # Filtrar documentos relevantes
            relevant_docs = [
                doc for doc, grade in state["graded_docs"] 
                if grade.binary == "yes"
            ]
            
            # Construir contexto
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
            else:
                context = "No hay información relevante en el contexto."
            
            # Generar respuesta
            chain = self.components.chains["answer_generation"]
            response = chain.invoke({
                "context": context,
                "query": state["query"]
            })
            
            answer = response.content
            
            logger.info(f"Respuesta generada: {len(answer)} caracteres")
            
            return {
                "answer": answer,
                "messages": [
                    HumanMessage(content=state["query"]),
                    AIMessage(content=answer)
                ]
            }
            
        except Exception as e:
            logger.error(f"Error en generate_answer: {e}")
            return {
                "answer": f"Error generando respuesta: {str(e)}",
                "messages": [AIMessage(content=f"Error: {str(e)}")]
            }
    
    def evaluate_answer(self, state: AgentState) -> dict:
        """
        Nodo 7: Evaluar calidad de respuesta (opcional, para reflection).
        
        Args:
            state: Estado actual
        
        Returns:
            Actualización del estado
        """
        logger.info("Nodo: evaluate_answer")
        
        try:
            chain = self.components.chains["answer_evaluation"]
            evaluation = chain.invoke({
                "query": state["query"],
                "answer": state["answer"]
            })
            
            logger.info(
                f"Evaluación: precision={evaluation.precision}, "
                f"completeness={evaluation.completeness}, "
                f"clarity={evaluation.clarity}"
            )
            
            return {
                "metadata": {
                    **state.get("metadata", {}),
                    "answer_evaluation": evaluation.model_dump()
                },
                "messages": [
                    AIMessage(
                        content=f"Evaluación: {evaluation.overall_score:.2f}/1.0\n"
                               f"Feedback: {evaluation.feedback}"
                    )
                ]
            }
            
        except Exception as e:
            logger.error(f"Error en evaluate_answer: {e}")
            return {"messages": [AIMessage(content=f"Error evaluando: {str(e)}")]}


# =============================================================================
# AGENTE PRINCIPAL
# =============================================================================

class AgenticRAG:
    """
    Agente RAG de producción con LangGraph.
    
    Este agente implementa el patrón Agentic RAG con:
    - Generación de query optimizada
    - Retrieval con evaluación de documentos
    - Transformación de query si es necesario
    - Generación de respuesta basada en contexto
    - Evaluación de calidad (opcional)
    
    Ejemplo de uso:
        config = AgentConfig()
        agent = AgenticRAG(config)
        result = agent.invoke("¿Qué es una tutela?")
        print(result["answer"])
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Inicializar agente RAG.
        
        Args:
            config: Configuración del agente (opcional, usa defaults si es None)
        """
        self.config = config or AgentConfig()
        
        # Configurar logging
        logging.getLogger().setLevel(getattr(logging, self.config.log_level))
        
        logger.info("Inicializando AgenticRAG")
        
        # Inicializar componentes
        self.components = RAGAgentComponents(self.config)
        
        # Inicializar nodos
        self.nodes = AgentNodes(self.components)
        
        # Construir grafo
        self.graph = self._build_graph()
        
        logger.info("AgenticRAG inicializado correctamente")
    
    def _build_graph(self) -> StateGraph:
        """
        Construir grafo del agente.
        
        Returns:
            Grafo compilado
        """
        logger.info("Construyendo grafo del agente")
        
        # Crear builder
        builder = StateGraph(AgentState)
        
        # Agregar nodos
        builder.add_node("generate_query", self.nodes.generate_query)
        builder.add_node("retrieve_documents", self.nodes.retrieve_documents)
        builder.add_node("grade_documents", self.nodes.grade_documents)
        builder.add_node("transform_query", self.nodes.transform_query)
        builder.add_node("generate_answer", self.nodes.generate_answer)
        builder.add_node("evaluate_answer", self.nodes.evaluate_answer)
        
        # Agregar edges
        builder.add_edge(START, "generate_query")
        builder.add_edge("generate_query", "retrieve_documents")
        builder.add_edge("retrieve_documents", "grade_documents")
        
        # Edge condicional para verificar documentos
        builder.add_conditional_edges(
            "grade_documents",
            self.nodes.check_documents,
            {
                "generate_answer": "generate_answer",
                "transform_query": "transform_query"
            }
        )
        
        # Bucle de transformación
        builder.add_edge("transform_query", "retrieve_documents")
        
        # Evaluación opcional (se puede habilitar/deshabilitar)
        builder.add_edge("generate_answer", "evaluate_answer")
        builder.add_edge("evaluate_answer", END)
        
        # Compilar con checkpointer para persistencia
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        
        logger.info("Grafo del agente compilado exitosamente")
        
        return graph
    
    def invoke(
        self, 
        query: str, 
        thread_id: str = "default",
        enable_evaluation: bool = True
    ) -> dict:
        """
        Invocar agente con una query.
        
        Args:
            query: Query del usuario
            thread_id: ID de hilo para persistencia
            enable_evaluation: Si habilitar evaluación de respuesta
        
        Returns:
            Estado final con respuesta
        """
        logger.info(f"Invocando agente con query: {query[:100]}...")
        
        # Configurar estado inicial
        initial_state = {
            "messages": [],
            "query": query,
            "current_query": query,
            "documents": [],
            "graded_docs": [],
            "answer": None,
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "metadata": {}
        }
        
        # Configurar thread
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        try:
            # Invocar grafo
            result = self.graph.invoke(initial_state, config=config)
            
            logger.info(f"Agente completado exitosamente")
            
            return result
            
        except Exception as e:
            logger.error(f"Error invocando agente: {e}")
            raise
    
    def stream(
        self, 
        query: str, 
        thread_id: str = "default"
    ):
        """
        Invocar agente con streaming.
        
        Args:
            query: Query del usuario
            thread_id: ID de hilo para persistencia
        
        Yields:
            Actualizaciones del estado
        """
        logger.info(f"Invocando agente con streaming: {query[:100]}...")
        
        initial_state = {
            "messages": [],
            "query": query,
            "current_query": query,
            "documents": [],
            "graded_docs": [],
            "answer": None,
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "metadata": {}
        }
        
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        try:
            for chunk in self.graph.stream(initial_state, config=config):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error en streaming: {e}")
            raise
    
    def get_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """
        Obtener historial de un thread.
        
        Args:
            thread_id: ID de thread
        
        Returns:
            Lista de mensajes
        """
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        try:
            state = self.graph.get_state(config)
            return state.values.get("messages", [])
            
        except Exception as e:
            logger.error(f"Error obteniendo historial: {e}")
            return []


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def main():
    """Ejemplo de uso del agente AgenticRAG."""
    
    print("=" * 80)
    print("MÓDULO 11: PATRONES AVANZADOS DE AGENTES")
    print("=" * 80)
    print("\nInicializando AgenticRAG...")
    
    # Configurar agente
    config = AgentConfig(
        llm_model="gemini-2.0-flash-exp",
        retrieval_k=3,
        max_iterations=3,
        relevance_threshold=0.7
    )
    
    # Crear agente
    agent = AgenticRAG(config)
    
    # Query de ejemplo
    query = "¿Qué es una acción de tutela?"
    
    print(f"\n📝 Query: {query}")
    print("-" * 80)
    
    # Invocar agente
    try:
        result = agent.invoke(query, thread_id="demo-001")
        
        print(f"\n✅ Respuesta:")
        print(f"{result['answer']}")
        
        print(f"\n📊 Metadata:")
        if "answer_evaluation" in result.get("metadata", {}):
            eval_data = result["metadata"]["answer_evaluation"]
            print(f"  Precisión: {eval_data.get('precision', 'N/A')}")
            print(f"  Completitud: {eval_data.get('completeness', 'N/A')}")
            print(f"  Claridad: {eval_data.get('clarity', 'N/A')}")
        
        print(f"\n💬 Historial:")
        history = agent.get_history("demo-001")
        for msg in history[-4:]:  # Últimos 4 mensajes
            emoji = "👤" if isinstance(msg, HumanMessage) else "🤖"
            print(f"  {emoji} {msg.content[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error en main: {e}")
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
