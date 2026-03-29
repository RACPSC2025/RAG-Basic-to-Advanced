"""
01_conexion_llm.py
Conexión básica con Google Gemini usando LangChain

Objetivo: Entender los parámetros de configuración del modelo
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def crear_modelo_basico():
    """Crear un modelo con configuración básica"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
    )


def crear_modelo_avanzado():
    """Crear un modelo con configuración avanzada"""
    return ChatGoogleGenerativeAI(
        # Modelo: gemini-2.0-flash-exp es gratuito y rápido
        model="gemini-2.0-flash-exp",
        
        # Temperature: 0.0 (determinista) a 2.0 (creativo)
        temperature=0.5,
        
        # max_output_tokens: Límite de tokens en la respuesta
        max_output_tokens=None,
        
        # top_p: Nucleus sampling
        top_p=0.95,
        
        # timeout: Tiempo máximo de espera en segundos
        timeout=None,
        
        # max_retries: Reintentos ante fallos
        max_retries=3,
    )


def probar_modelo():
    """Probar el modelo con diferentes prompts"""
    
    llm = crear_modelo_avanzado()
    
    # Prompt 1: Pregunta factual
    print("=" * 60)
    print("PROMPT 1: Pregunta Factual")
    print("=" * 60)
    response = llm.invoke("¿Cuál es la capital de Colombia?")
    print(f"Pregunta: ¿Cuál es la capital de Colombia?")
    print(f"Respuesta: {response.content}\n")
    
    # Prompt 2: Pregunta creativa
    print("=" * 60)
    print("PROMPT 2: Pregunta Creativa")
    print("=" * 60)
    response = llm.invoke("Inventa un nombre para una startup de IA que usa animales")
    print(f"Pregunta: Inventa un nombre para una startup de IA que usa animales")
    print(f"Respuesta: {response.content}\n")
    
    # Prompt 3: Instrucción compleja
    print("=" * 60)
    print("PROMPT 3: Instrucción Compleja")
    print("=" * 60)
    prompt = """
    Eres un asistente legal experto en derecho colombiano.
    Responde de forma clara y concisa.
    
    Pregunta: ¿Qué es una acción de tutela?
    """
    response = llm.invoke(prompt)
    print(f"Respuesta: {response.content}\n")


def comparar_temperaturas():
    """Comparar respuestas con diferentes temperaturas"""
    
    print("=" * 60)
    print("COMPARACIÓN DE TEMPERATURAS")
    print("=" * 60)
    
    pregunta = "Da 3 ideas para usar IA en un bufete de abogados"
    
    for temp in [0.0, 0.5, 1.0]:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=temp,
        )
        
        print(f"\n--- Temperature: {temp} ---")
        response = llm.invoke(pregunta)
        print(f"{response.content[:200]}...\n")


def manejar_errores():
    """Demostrar manejo de errores"""
    
    from google.api_core.exceptions import ResourceExhausted
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        max_retries=3,
    )
    
    try:
        response = llm.invoke("Hola")
        print(f"✅ Success: {response.content[:50]}...")
        
    except ResourceExhausted:
        print("❌ Error: Rate limit excedido. Espera un momento.")
        
    except Exception as e:
        print(f"❌ Error inesperado: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Ejecutar pruebas
    probar_modelo()
    comparar_temperaturas()
    manejar_errores()
