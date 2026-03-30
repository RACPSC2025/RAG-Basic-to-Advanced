"""
Módulo de LLM - AWS Bedrock Integration

Proporciona funciones para crear y configurar el modelo de lenguaje
usando AWS Bedrock (Amazon Nova Lite).

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
"""

import logging
from typing import Optional
from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel

from .config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AWS_REGION,
    LLM_MODEL_ID,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def create_llm(
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Crear una instancia de LLM usando AWS Bedrock.

    Args:
        model_id: ID del modelo (ARN para inference profiles)
        temperature: Temperatura del modelo (0.0-1.0)
        max_tokens: Máximo de tokens en la respuesta
        provider: Proveedor del modelo (amazon, anthropic, meta, etc.)
        **kwargs: Argumentos adicionales para ChatBedrock

    Returns:
        Instancia de ChatBedrock configurada

    Ejemplo:
        >>> llm = create_llm(temperature=0.5)
        >>> response = llm.invoke("¿Qué es una tutela?")
    """
    model_id = model_id or LLM_MODEL_ID
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens or LLM_MAX_TOKENS
    provider = provider or LLM_PROVIDER

    logger.debug(
        f"Creando LLM: model={model_id}, temp={temperature}, max_tokens={max_tokens}"
    )

    # Configurar credenciales AWS
    bedrock_config = {
        "model_id": model_id,
        "provider": provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "region_name": AWS_REGION,
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    }

    # Agregar session token si existe (para credenciales temporales)
    if AWS_SESSION_TOKEN:
        bedrock_config["aws_session_token"] = AWS_SESSION_TOKEN

    # Agregar kwargs adicionales
    bedrock_config.update(kwargs)

    llm = ChatBedrock(**bedrock_config)

    logger.info(f"LLM creado exitosamente: {model_id}")

    return llm


def get_default_llm() -> BaseChatModel:
    """
    Obtener el LLM por defecto con configuración estándar.

    Returns:
        Instancia de ChatBedrock con configuración por defecto
    """
    return create_llm()


def get_llm_for_task(task_type: str) -> BaseChatModel:
    """
    Obtener un LLM configurado para un tipo de tarea específica.

    Args:
        task_type: Tipo de tarea ("chat", "analysis", "summary", "extraction")

    Returns:
        Instancia de ChatBedrock configurada para la tarea
    """
    configs = {
        "chat": {
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        "analysis": {
            "temperature": 0.1,  # Más determinista para análisis legal
            "max_tokens": 2048,
        },
        "summary": {
            "temperature": 0.2,
            "max_tokens": 512,
        },
        "extraction": {
            "temperature": 0.0,  # Completamente determinista
            "max_tokens": 1024,
        },
    }

    config = configs.get(task_type, configs["chat"])

    logger.debug(f"Obteniendo LLM para tarea '{task_type}': {config}")

    return create_llm(**config)


# Singleton para el LLM por defecto
_default_llm: Optional[BaseChatModel] = None


def get_cached_llm() -> BaseChatModel:
    """
    Obtener o crear el LLM cacheado (singleton).

    Útil para evitar crear múltiples instancias del mismo modelo.

    Returns:
        Instancia de ChatBedrock cacheada
    """
    global _default_llm

    if _default_llm is None:
        logger.debug("Creando LLM cacheado por primera vez")
        _default_llm = get_default_llm()

    return _default_llm


def clear_llm_cache():
    """
    Limpiar el cache del LLM.

    Útil para testing o cuando se cambian credenciales.
    """
    global _default_llm
    _default_llm = None
    logger.debug("Cache de LLM limpiado")
