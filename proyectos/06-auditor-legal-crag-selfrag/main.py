"""
Auditor Legal Inteligente (CRAG + Self-RAG) - Proyecto 6
Punto de entrada principal.

Migrado a AWS Bedrock - 2026-03-30
"""

import sys
import os

# Añadir el directorio actual al path para importaciones relativas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import create_auditor_graph

def main():
    print("\n" + "=" * 70)
    print("⚖️  BIENVENIDO AL AUDITOR LEGAL INTELIGENTE (CRAG + SELF-RAG)")
    print("=" * 70)
    print("Este sistema validará su consulta legal colombiana con máxima precisión.")
    print("Tecnología: AWS Bedrock (Amazon Nova Lite + Titan Embeddings v2)")
    print("=" * 70)

    # 1. Cargar el Grafo
    print("\n📦 Cargando grafo del auditor...")
    app = create_auditor_graph()
    print("✅ Grafo cargado exitosamente")

    # 2. Configurar la consulta inicial
    query = "¿Cuál es el procedimiento legal para una tutela de salud en Colombia?"
    print(f"\n📝 Consulta de prueba: {query}")
    
    inputs = {
        "question": query,
        "retries": 0,
        "steps": [],
        "documents": []
    }

    # 3. Ejecutar el Auditor (Streaming de pasos)
    print("\n🔍 Ejecutando auditoría...\n")
    
    final_generation = None
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"--- [FASE: {key.upper()}] ---")
            if "steps" in value:
                print(f"  Historial de auditoría: {value['steps']}")
            if "generation" in value:
                final_generation = value["generation"]
                print(f"  Respuesta generada: {value['generation'][:200]}...")
            if "is_relevant" in value:
                print(f"  Documentos relevantes: {value['is_relevant']}")
            if "hallucination" in value:
                print(f"  Alucinación detectada: {value['hallucination']}")
            if "answer_relevant" in value:
                print(f"  Respuesta útil: {value['answer_relevant']}")
            print()

    print("\n" + "=" * 70)
    print("✅ AUDITORÍA FINALIZADA")
    print("=" * 70)
    
    if final_generation:
        print(f"\n📄 Respuesta final:\n{final_generation}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
