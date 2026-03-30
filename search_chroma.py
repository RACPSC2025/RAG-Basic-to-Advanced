import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from proyectos.Rag_Legal.ingestor import get_vector_store

print("Buscando '2.2.1.1.7' en ChromaDB...")
try:
    vector_store = get_vector_store()
    query = "2.2.1.1.7"
    docs = vector_store.similarity_search(query, k=5)
    print(f"Encontrados {len(docs)} docs:")
    for i, d in enumerate(docs, 1):
        print(f"\n--- Doc {i} ---")
        print(d.page_content[:400])
except Exception as e:
    print(f"Error: {e}")
