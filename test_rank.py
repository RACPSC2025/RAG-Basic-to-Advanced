import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from proyectos.Rag_Legal.ingestor import get_vector_store

print("Buscando '2.2.1.1.7' en ChromaDB rankeado...")
try:
    vector_store = get_vector_store()
    query = "ARTÍCULO 2.2.1.1.7. Sanción disciplinaria"
    docs_with_scores = vector_store.similarity_search_with_score(query, k=50)
    
    found_idx = -1
    for i, (doc, score) in enumerate(docs_with_scores):
        if "2.2.1.1.7" in doc.page_content:
            found_idx = i
            print(f"✅ ¡Encontrado en la posición {i+1} (score: {score})!")
            break
            
    if found_idx == -1:
        print("❌ '2.2.1.1.7' no aparece en el Top 50!")
        
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
