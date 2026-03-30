import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from proyectos.Rag_Legal.ingestor import get_vector_store

print("Buscando '2.2.1.1.7' exacto en ChromaDB via GET...")
try:
    vector_store = get_vector_store()
    results = vector_store.get()
    
    contents = results.get("documents", [])
    found_count = 0
    for content in contents:
        if "2.2.1.1.7" in content:
            found_count += 1
            print(f"\n--- MATCH {found_count} ---")
            print(content[:500])
            
    if found_count == 0:
        print("❌ El texto '2.2.1.1.7' NO existe en ningún documento de ChromaDB.")
    else:
        print(f"\n✅ Se encontraron {found_count} ocurrencias del texto '2.2.1.1.7' en ChromaDB.")

except Exception as e:
    print(f"Error: {e}")
