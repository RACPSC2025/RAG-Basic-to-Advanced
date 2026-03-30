"""
Módulo de Memoria - Chat Memory Management

Implementa memoria de conversación a corto y largo plazo para el chatbot.
- Corto plazo: Historial reciente de mensajes (ventana deslizante)
- Largo plazo: Resumen de la conversación generado por el LLM

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from .config import (
    MEMORY_SHORT_TERM_K,
    MEMORY_LONG_TERM_ENABLED,
)
from .llm import get_default_llm

logger = logging.getLogger(__name__)


class ChatMemory:
    """
    Gestor de memoria para conversaciones del chatbot.

    Características:
    - Memoria a corto plazo: Últimos K mensajes
    - Memoria a largo plazo: Resumen automático de la conversación
    - Métodos para agregar, obtener y limpiar mensajes

    Ejemplo:
        >>> memory = ChatMemory()
        >>> memory.add_message("¿Qué es una tutela?", "Una tutela es...")
        >>> history = memory.get_history()
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        k: Optional[int] = None,
        enable_long_term: Optional[bool] = None,
    ):
        """
        Inicializar la memoria del chat.

        Args:
            llm: Instancia del LLM para resúmenes (opcional)
            k: Número de mensajes a recordar (ventana deslizante)
            enable_long_term: Si habilitar memoria a largo plazo
        """
        self.k = k if k is not None else MEMORY_SHORT_TERM_K
        self.enable_long_term = enable_long_term if enable_long_term is not None else MEMORY_LONG_TERM_ENABLED

        self.llm = llm or get_default_llm()

        # Historial de mensajes (corto plazo)
        self._messages: List[BaseMessage] = []

        # Resumen de largo plazo
        self._long_term_summary: str = ""
        self._summary_turn_count: int = 0

        # Contadores
        self.turn_count: int = 0
        self.total_messages: int = 0

        # Prompt para generar resúmenes
        self._summary_prompt = """Resume la siguiente conversación en un párrafo conciso (máximo 100 palabras).
El resumen debe capturar los temas principales discutidos y cualquier conclusión importante.

Conversación:
{conversation}

Resumen:"""

        logger.debug(f"ChatMemory inicializada: k={self.k}, long_term={self.enable_long_term}")

    def add_message(self, user_input: str, assistant_response: str):
        """
        Agregar un par de mensajes (usuario + asistente) al historial.

        Args:
            user_input: Mensaje del usuario
            assistant_response: Respuesta del asistente
        """
        logger.debug(f"Agregando mensaje: turn={self.turn_count + 1}")

        # Agregar mensajes
        self._messages.append(HumanMessage(content=user_input))
        self._messages.append(AIMessage(content=assistant_response))

        self.turn_count += 1
        self.total_messages += 2

        # Mantener ventana deslizante (solo últimos 2k mensajes)
        max_messages = self.k * 2
        if len(self._messages) > max_messages:
            removed = self._messages[:len(self._messages) - max_messages]
            self._messages = self._messages[-max_messages:]
            logger.debug(f"Memoria recortada: {len(removed)} mensajes eliminados")

        # Actualizar resumen de largo plazo cada N turnos
        if self.enable_long_term and self.turn_count % 5 == 0:
            self._update_long_term_summary()

    def _update_long_term_summary(self):
        """
        Actualizar el resumen de largo plazo usando el LLM.
        """
        if not self._messages:
            return

        try:
            # Construir conversación actual
            conversation_text = self._format_messages_for_summary()

            # Si ya hay resumen, incluirlo en el prompt
            if self._long_term_summary:
                prompt_content = f"""Contexto previo de la conversación:
{_long_term_summary}

Nuevos mensajes:
{conversation_text}

Resume toda la conversación (contexto previo + nuevos mensajes) en un párrafo conciso:"""
            else:
                prompt_content = self._summary_prompt.format(conversation=conversation_text)

            # Generar resumen
            response = self.llm.invoke(prompt_content)
            self._long_term_summary = response.content
            self._summary_turn_count = self.turn_count

            logger.debug(f"Resumen actualizado: turn={self.turn_count}, len={len(self._long_term_summary)}")

        except Exception as e:
            logger.warning(f"Error actualizando resumen de largo plazo: {e}")

    def _format_messages_for_summary(self) -> str:
        """
        Formatear mensajes para el resumen.

        Returns:
            String con los mensajes formateados
        """
        lines = []
        for msg in self._messages[-10:]:  # Últimos 10 mensajes
            role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
            content = msg.content[:200]  # Truncar si es muy largo
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def get_history(self) -> List[BaseMessage]:
        """
        Obtener el historial de mensajes para pasar al LLM.

        Returns:
            Lista de mensajes de LangChain
        """
        # Incluir resumen de largo plazo si existe
        if self.enable_long_term and self._long_term_summary:
            summary_message = SystemMessage(
                content=f"Resumen de la conversación anterior: {self._long_term_summary}"
            )
            return [summary_message] + self._messages

        return self._messages.copy()

    def get_context(self) -> str:
        """
        Obtener el contexto actual de la conversación como texto.

        Returns:
            String con el contexto formateado
        """
        lines = []

        if self.enable_long_term and self._long_term_summary:
            lines.append(f"[Contexto previo: {self._long_term_summary}]")

        for msg in self._messages[-6:]:  # Últimos 6 mensajes
            role = "Tú" if isinstance(msg, HumanMessage) else "Asistente"
            lines.append(f"{role}: {msg.content[:150]}")

        return "\n".join(lines)

    def clear(self):
        """
        Limpiar toda la memoria (corto y largo plazo).
        """
        logger.info("Limpiando memoria de conversación")

        self._messages.clear()
        self._long_term_summary = ""
        self._summary_turn_count = 0
        self.turn_count = 0
        self.total_messages = 0

        logger.debug("Memoria limpiada exitosamente")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la memoria.

        Returns:
            Diccionario con estadísticas
        """
        return {
            "turn_count": self.turn_count,
            "total_messages": self.total_messages,
            "short_term_messages": len(self._messages),
            "summary_length": len(self._long_term_summary),
            "summary_turn_count": self._summary_turn_count,
            "long_term_enabled": self.enable_long_term,
            "window_size": self.k,
        }

    def get_messages(self) -> List[BaseMessage]:
        """
        Obtener la lista completa de mensajes.

        Returns:
            Lista de mensajes
        """
        return self._messages.copy()

    def set_messages(self, messages: List[BaseMessage]):
        """
        Establecer manualmente la lista de mensajes.

        Útil para restaurar conversaciones guardadas.

        Args:
            messages: Lista de mensajes a establecer
        """
        self._messages = messages.copy()
        self.turn_count = len(messages) // 2
        self.total_messages = len(messages)

        logger.debug(f"Mensajes establecidos: {len(messages)} mensajes, {self.turn_count} turnos")

    def __len__(self) -> int:
        """Retornar número de mensajes en memoria."""
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ChatMemory(turns={self.turn_count}, messages={len(self._messages)}, summary={len(self._long_term_summary)} chars)"
