"""
Test de Retrieval RAG Fenix
Verifica qué devuelve ChromaDB y cómo evalúa el Grader LLM.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from proyectos.Rag_Legal.ingestor import get_vector_store
from proyectos.Rag_Legal.nodes import _GRADE_PROMPT
from proyectos.Rag_Legal.config import get_llm
from pydantic import BaseModel, Field

import builtins

with open("test_retrieval_output.txt", "w", encoding="utf-8") as f:
    def print(*args, **kwargs):
        kwargs["file"] = f
        builtins.print(*args, **kwargs)

    # 1. Verificar ChromaDB
    print("1. Conectando a ChromaDB...")
    try:
        vector_store = get_vector_store()
        count = len(vector_store.get()["ids"])
        print(f"[OK] ChromaDB tiene {count} documentos indexados.")
        if count == 0:
            print("[ERR] ChromaDB está vacío. La ingesta falló o no se ha completado!")
            sys.exit(1)
    except Exception as e:
        print(f"[ERR] Error al conectar a ChromaDB: {e}")
        sys.exit(1)

    # 2. Retrieval
    query = "ARTÍCULO 2.2.1.1.7. Sanción disciplinaria"
    print(f"\n2. Buscando query: '{query}'")
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    if not docs:
        print("[ERR] No se encontraron documentos similares.")
        sys.exit(1)

    print(f"[OK] Se recuperaron {len(docs)} documentos:")
    for i, d in enumerate(docs, 1):
        print(f"\n--- Doc {i} (Source: {d.metadata.get('source', 'Unknown')}) ---")
        print(f"{d.page_content[:400]}...")

    # 3. Grader
    print("\n3. Evaluando relevancia (Grader Node)...")
    class GradeOutput(BaseModel):
        score: str = Field(description="'si' o 'no'")
        razon: str = Field(description="Explicación")

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(GradeOutput)
        chain = _GRADE_PROMPT | structured_llm
        
        for i, doc in enumerate(docs, 1):
            res = chain.invoke({"question": query, "document": doc.page_content})
            print(f"Doc {i} -> Score: {res.score} | Razon: {res.razon}")
    except Exception as e:
        print(f"[ERR] Error en el Grader LLM: {e}")
