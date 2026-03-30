"""
Chatbot Legal Básico - Paquete Principal

Este paquete implementa un chatbot legal con:
- LLM AWS Bedrock (Amazon Nova Lite)
- Memoria de conversación (corto y largo plazo)
- Human in the Loop para aprobación de respuestas críticas

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
"""

from .chatbot import LegalChatbot, create_chatbot, get_default_chatbot
from .config import (
    APP_NAME,
    APP_VERSION,
    AWS_REGION,
    LLM_MODEL_ID,
    LLM_TEMPERATURE,
    HITL_ENABLED,
    HITL_CONFIDENCE_THRESHOLD,
)
from .llm import create_llm, get_default_llm, get_cached_llm
from .memory import ChatMemory
from .human_in_loop import HumanApproval

__all__ = [
    # Chatbot
    "LegalChatbot",
    "create_chatbot",
    "get_default_chatbot",
    # Configuración
    "APP_NAME",
    "APP_VERSION",
    "AWS_REGION",
    "LLM_MODEL_ID",
    "LLM_TEMPERATURE",
    "HITL_ENABLED",
    "HITL_CONFIDENCE_THRESHOLD",
    # LLM
    "create_llm",
    "get_default_llm",
    "get_cached_llm",
    # Memoria
    "ChatMemory",
    # Human in the Loop
    "HumanApproval",
]
