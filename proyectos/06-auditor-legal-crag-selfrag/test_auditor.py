"""
Test del Proyecto 6 - Auditor Legal CRAG + Self-RAG
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent.graph import create_auditor_graph
from src.config import Config

def test_config():
    """Verificar configuración."""
    print("=" * 70)
    print("TEST DE CONFIGURACION")
    print("=" * 70)
    print(f"LLM Model: {Config.LLM_MODEL_ID}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL_ID}")
    print(f"RPM: {Config.RPM}")
    print(f"Request Delay: {Config.REQUEST_DELAY:.2f}s")
    print(f"Max Retries: {Config.MAX_RETRIES}")
    print(f"AWS Access Key: {'CONFIGURADA' if Config.AWS_ACCESS_KEY_ID else 'NO CONFIGURADA'}")
    print()

def test_graph_creation():
    """Verificar creacion del grafo."""
    print("=" * 70)
    print("TEST DE CREACION DEL GRAFO")
    print("=" * 70)
    try:
        graph = create_auditor_graph()
        print("✅ Grafo creado exitosamente")
        print(f"   Tipo: {type(graph)}")
        return graph
    except Exception as e:
        print(f"❌ Error creando grafo: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_basic_query(graph):
    """Probar consulta basica."""
    print("\n" + "=" * 70)
    print("TEST DE CONSULTA BASICA")
    print("=" * 70)
    
    if graph is None:
        print("❌ No hay grafo disponible")
        return
    
    inputs = {
        "question": "Que es una tutela en Colombia?",
        "retries": 0,
        "steps": [],
        "documents": []
    }
    
    try:
        print("Ejecutando consulta...")
        for output in graph.stream(inputs):
            for key, value in output.items():
                print(f"  [{key.upper()}]")
                if "generation" in value:
                    print(f"  Respuesta: {value['generation'][:200]}...")
                if "steps" in value:
                    print(f"  Steps: {value['steps']}")
        print("\n✅ Consulta completada")
    except Exception as e:
        print(f"❌ Error en consulta: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("\n" + "=" * 70)
    print("TEST COMPLETO - PROYECTO 6: AUDITOR LEGAL (CRAG + SELF-RAG)")
    print("=" * 70)
    
    # Test 1: Config
    test_config()
    
    # Test 2: Creacion del grafo
    graph = test_graph_creation()
    
    # Test 3: Consulta basica (si hay grafo)
    if graph:
        test_basic_query(graph)
    
    print("\n" + "=" * 70)
    print("TEST FINALIZADO")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR CRITICO: {e}")
        import traceback
        traceback.print_exc()
