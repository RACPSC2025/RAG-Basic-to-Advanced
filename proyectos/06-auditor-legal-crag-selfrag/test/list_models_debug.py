import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("\n--- 🔍 ESCANEANDO MODELOS DE EMBEDDINGS DISPONIBLES ---")

try:
    models = genai.list_models()
    found = False
    for m in models:
        if 'embedContent' in m.supported_generation_methods:
            print(f"✅ Modelo Encontrado: {m.name}")
            print(f"   Descripción: {m.description}")
            print(f"   Input Token Limit: {m.input_token_limit}\n")
            found = True
    
    if not found:
        print("❌ No se encontraron modelos que soporten 'embedContent' en esta API Key.")
        
except Exception as e:
    print(f"❌ Error al listar modelos: {e}")
