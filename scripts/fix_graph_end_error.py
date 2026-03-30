"""
Corrección de Error: KeyError '__end__'

Este script arregla el bug en graph.py donde se usa "END" (string) 
en lugar de END (constante de langgraph).

Ejecutar:
    python scripts/fix_graph_end_error.py
"""

from pathlib import Path

def fix_graph_file():
    """Corregir el archivo graph.py."""
    
    graph_path = Path("proyectos/06-auditor-legal-crag-selfrag/src/agent/graph.py")
    
    if not graph_path.exists():
        print(f"❌ Archivo no encontrado: {graph_path}")
        return False
    
    print(f"📄 Leyendo: {graph_path}")
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Verificar si ya está corregido
    if 'return END  # Finalizar' in content or 'return END\n' in content:
        print("✅ El archivo YA está corregido")
        return False
    
    # Corregir el error específico
    # Buscar: return "END" en la función decide_answer_quality
    fixes = [
        ('return "END"  # Finalizar', 'return END  # Finalizar'),
        ('return "END"', 'return END'),
    ]
    
    changes_made = 0
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            changes_made += 1
            print(f"  ✓ Corregido: {old} → {new}")
    
    if changes_made > 0:
        with open(graph_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✅ CORRECCIÓN COMPLETADA")
        print(f"📝 Archivo actualizado: {graph_path}")
        print(f"\n🔄 Ahora prueba de nuevo:")
        print(f"   python proyectos/06-auditor-legal-crag-selfrag/cli.py")
        return True
    else:
        print(f"\n⚠️ No se encontraron los patrones esperados")
        print(f"   Revisa manualmente: {graph_path}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🔧 CORRECCIÓN: KeyError '__end__'")
    print("=" * 80)
    fix_graph_file()
    print("=" * 80)
