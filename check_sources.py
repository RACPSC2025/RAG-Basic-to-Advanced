import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from proyectos.Rag_Legal.ingestor import get_vector_store

print("Analizando fuentes en ChromaDB...")
try:
    vector_store = get_vector_store()
    results = vector_store.get()
    
    metadatas = results.get("metadatas", [])
    sources = set()
    for m in metadatas:
        sources.add(m.get("source", "Unknown"))
    
    print(f"Fuentes encontradas: {sources}")
    print(f"Total documentos: {len(metadatas)}")
    
    # Buscar especificamente por el archivo sample.pdf
    sample_docs = [m for m in metadatas if m.get("source") == "sample.pdf"]
    print(f"Documentos de 'sample.pdf': {len(sample_docs)}")

except Exception as e:
    print(f"Error: {e}")
