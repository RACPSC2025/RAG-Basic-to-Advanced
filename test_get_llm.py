"""
Test de inicializacion de get_llm().
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

from proyectos.Rag_Legal.config import get_llm

print("Intentando inicializar get_llm()...")
try:
    llm = get_llm()
    print("✅ LLM inicializado correctamente.")
    print(f"Tipo del LLM: {type(llm)}")
    print(f"Provider: {getattr(llm, 'provider', 'N/A')}")
except Exception as e:
    print(f"❌ Error al inicializar: {type(e).__name__}: {e}")
