"""
Chatbot Legal Básico - Módulo Principal

Este módulo implementa el chatbot legal completo integrando:
- LLM (Google Gemini)
- Memoria (corto y largo plazo)
- Human in the Loop
- Logging y manejo de errores

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
License: MIT
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from .config import (
    APP_NAME,
    APP_VERSION,
    LOG_LEVEL,
    HITL_CONFIDENCE_THRESHOLD
)
from .llm import get_default_llm, create_llm
from .memory import ChatMemory
from .human_in_loop import HumanApproval

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class LegalChatbot:
    """
    Chatbot Legal Básico con memoria y aprobación humana.
    
    Este chatbot responde consultas legales básicas usando
    el conocimiento del LLM, manteniendo contexto de conversación
    y solicitando aprobación humana para respuestas críticas.
    
    Ejemplo de uso:
        >>> chatbot = LegalChatbot()
        >>> response = chatbot.chat("¿Qué es una tutela?")
        >>> print(response["respuesta"])
    """
    
    def __init__(self, llm=None, enable_hitl: bool = True):
        """
        Inicializar el Chatbot Legal.
        
        Args:
            llm: Instancia del LLM (opcional, crea una por defecto si es None)
            enable_hitl: Si habilitar Human in the Loop
        """
        logger.info(f"Inicializando {APP_NAME} v{APP_VERSION}")
        
        # Inicializar LLM
        self.llm = llm or get_default_llm()
        logger.debug("LLM inicializado")
        
        # Inicializar memoria
        self.memory = ChatMemory(llm=self.llm)
        logger.debug("Memoria inicializada")
        
        # Inicializar Human in the Loop
        self.hitl = HumanApproval(enabled=enable_hitl)
        logger.debug(f"HITL inicializado: enabled={enable_hitl}")
        
        # Prompt template para el chatbot
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}")
        ])
        logger.debug("Prompt template configurado")
        
        # Chain principal
        self.chain = self.prompt | self.llm
        logger.debug("Chain principal configurado")
        
        logger.info(f"{APP_NAME} inicializado exitosamente")
    
    def _get_system_prompt(self) -> str:
        """
        Obtener el system prompt para el chatbot.
        
        Returns:
            System prompt como string
        """
        return """Eres un asistente legal experto en derecho colombiano.

TU ROL:
- Responder consultas legales básicas de forma clara y precisa
- Citar artículos y normas cuando sea relevante
- Usar lenguaje técnico pero accesible
- Ser honesto cuando no sepas algo
- NO dar consejos legales vinculantes
- Recomendar consultar abogado para casos específicos

TUS LÍMITES:
- Solo responde preguntas legales
- No eres un abogado licenciado
- Tu conocimiento tiene fecha de corte
- Las leyes pueden cambiar, verifica vigencia

FORMATO DE RESPUESTA:
- Sé conciso pero completo
- Usa viñetas para listar elementos
- Cita artículos cuando los conozcas
- Incluye advertencias cuando sea necesario

RECUERDA:
- El usuario puede no tener conocimiento legal
- Explica términos técnicos
- Sé empático con situaciones difíciles
"""
    
    def generate_response(
        self, 
        query: str,
        use_memory: bool = True
    ) -> Dict:
        """
        Generar respuesta para una consulta legal.
        
        Args:
            query: Consulta del usuario
            use_memory: Si usar memoria de conversación
        
        Returns:
            Diccionario con:
            - respuesta: str
            - confidence: float (estimada)
            - requires_approval: bool
            - sources: list (si aplica)
        """
        logger.info(f"Generando respuesta para query: {query[:50]}...")
        
        try:
            # Preparar inputs para el chain
            chain_inputs = {"input": query}
            
            # Agregar historial si usa memoria
            if use_memory:
                chain_inputs["chat_history"] = self.memory.get_history()
            
            # Invocar chain
            response = self.chain.invoke(chain_inputs)
            
            # Extraer contenido de la respuesta
            response_text = response.content
            
            # Estimar confianza (basado en longitud y estructura)
            confidence = self._estimate_confidence(response_text, query)
            
            # Verificar si requiere aprobación humana
            requires_approval = (
                self.hitl.should_require_approval(
                    response=response_text,
                    confidence=confidence,
                    is_critical_topic=self.hitl.check_critical_topics(query)
                )
            )
            
            logger.debug(f"Confianza estimada: {confidence:.2f}")
            logger.debug(f"Requiere aprobación: {requires_approval}")
            
            return {
                "respuesta": response_text,
                "confidence": confidence,
                "requires_approval": requires_approval,
                "sources": [],  # No hay fuentes en este chatbot básico
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}", exc_info=True)
            
            return {
                "respuesta": f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}",
                "confidence": 0.0,
                "requires_approval": False,
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def _estimate_confidence(self, response: str, query: str) -> float:
        """
        Estimar confianza de la respuesta.
        
        Args:
            response: Respuesta generada
            query: Query original
        
        Returns:
            Confianza estimada (0-1)
        """
        confidence = 0.5  # Base
        
        # Respuestas muy cortas tienen menor confianza
        if len(response) < 50:
            confidence -= 0.2
        elif len(response) > 500:
            confidence += 0.1
        
        # Respuestas que admiten ignorancia tienen mayor confianza
        if any(phrase in response.lower() for phrase in [
            "no estoy seguro",
            "no tengo información",
            "te recomiendo consultar",
            "no puedo dar"
        ]):
            confidence += 0.2
        
        # Respuestas con estructura clara tienen mayor confianza
        if any(marker in response for marker in ["•", "-", "1.", "Artículo", "Ley"]):
            confidence += 0.1
        
        # Limitar a rango 0-1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def chat(
        self, 
        query: str,
        auto_approve: bool = False
    ) -> Dict:
        """
        Chatear con el chatbot (método principal de uso).
        
        Args:
            query: Consulta del usuario
            auto_approve: Si aprobar automáticamente (ignorar HITL)
        
        Returns:
            Diccionario con la respuesta final
        """
        logger.info(f"Chat iniciado: {query[:50]}...")
        
        # Generar respuesta
        response_data = self.generate_response(query)
        
        # Verificar si requiere aprobación
        if response_data["requires_approval"] and not auto_approve:
            logger.info("Solicitando aprobación humana")
            
            approved, feedback = self.hitl.request_approval(
                response=response_data["respuesta"],
                context=self.memory.get_context()
            )
            
            if not approved:
                if feedback and isinstance(feedback, str) and feedback != "Saltado por usuario":
                    # Usuario rechazó o editó
                    if feedback != "":
                        response_data["respuesta"] = feedback
                        response_data["edited_by_human"] = True
                    else:
                        response_data["respuesta"] = (
                            "Lo siento, pero no puedo proporcionar esa información. "
                            "Te recomiendo consultar con un abogado licenciado."
                        )
                        response_data["rejected_by_human"] = True
        
        # Agregar a memoria
        self.memory.add_message(query, response_data["respuesta"])
        
        logger.info(f"Chat completado, turnos totales: {self.memory.turn_count}")
        
        return response_data
    
    def reset_conversation(self):
        """
        Resetear la conversación (limpiar memoria).
        """
        logger.info("Reseteando conversación")
        self.memory.clear()
        logger.info("Conversación reseteada")
    
    def get_stats(self) -> Dict:
        """
        Obtener estadísticas de la conversación.
        
        Returns:
            Diccionario con estadísticas
        """
        memory_stats = self.memory.get_stats()
        
        return {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "turn_count": memory_stats["turn_count"],
            "short_term_messages": memory_stats["short_term_messages"],
            "long_term_summary_length": memory_stats.get("summary_length", 0),
            "hitl_enabled": self.hitl.enabled,
            "hitl_threshold": self.hitl.confidence_threshold
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Obtener historial de la conversación.
        
        Returns:
            Lista de dicts con historial
        """
        history = self.memory.get_history()
        
        return [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in history
        ]
    
    def __repr__(self):
        return f"LegalChatbot(app={APP_NAME}, version={APP_VERSION}, turns={self.memory.turn_count})"


# Función de conveniencia para crear chatbot
def create_chatbot(**kwargs) -> LegalChatbot:
    """
    Crear instancia de LegalChatbot.
    
    Args:
        **kwargs: Argumentos para LegalChatbot
    
    Returns:
        Instancia de LegalChatbot
    """
    return LegalChatbot(**kwargs)


# Chatbot singleton por defecto
_default_chatbot: Optional[LegalChatbot] = None


def get_default_chatbot() -> LegalChatbot:
    """
    Obtener o crear el chatbot por defecto.
    
    Returns:
        Instancia de LegalChatbot
    """
    global _default_chatbot
    
    if _default_chatbot is None:
        _default_chatbot = create_chatbot()
    
    return _default_chatbot
