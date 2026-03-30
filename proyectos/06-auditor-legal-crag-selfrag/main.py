import sys
import os

# Añadir el directorio actual al path para importaciones relativas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import create_auditor_graph

def main():
    print("\n--- ⚖️ BIENVENIDO AL AUDITOR LEGAL INTELIGENTE (CRAG + SELF-RAG) ---")
    print("Este sistema validará su consulta legal colombiana con máxima precisión.\n")
    
    # 1. Cargar el Grafo
    app = create_auditor_graph()
    
    # 2. Configurar la consulta inicial
    inputs = {
        "question": "¿Cuál es el procedimiento legal para una tutela de salud en Colombia?",
        "retries": 0,
        "steps": [],
        "documents": []
    }
    
    # 3. Ejecutar el Auditor (Streaming de pasos)
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"\n--- [FASE: {key.upper()}] ---")
            if "steps" in value:
                print(f"Historial de auditoría: {value['steps']}")
            if "generation" in value:
                print(f"Respuesta generada: {value['generation']}")
    
    print("\n--- ✅ AUDITORÍA FINALIZADA ---")

if __name__ == "__main__":
    main()
