"""
00_hello_langchain.py
Tu primer script con LangChain y Google Gemini

Objetivo: Verificar que la configuración funciona correctamente
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar variables de entorno
load_dotenv()


def main():
    """Función principal de prueba"""
    
    # 1. Verificar API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        print("❌ Error: GOOGLE_API_KEY no configurada en el archivo .env")
        print("   Sigue las instrucciones en docs/curso/00-configuracion/README.md")
        return
    
    print("✅ API Key encontrada")
    
    # 2. Inicializar el modelo
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash-lite"),  # Modelo gratuito recomendado
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    print(f"✅ Modelo inicializado: {llm.model}")
    
    # 3. Hacer una pregunta simple
    response = llm.invoke("Hola, ¿cómo estás? Responde brevemente.")
    
    print(f"\n🤖 Gemini dice: {response.content}")
    print("\n✅ ¡Configuración exitosa! LangChain + Gemini funcionan correctamente.")


if __name__ == "__main__":
    main()
