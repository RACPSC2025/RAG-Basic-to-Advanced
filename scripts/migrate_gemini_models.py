"""
Script de Migración de Modelos Google Gemini (v1beta)

Actualiza automáticamente los nombres de modelos antiguos a la nueva API v1beta.

Ejecutar:
    python scripts/migrate_gemini_models.py
"""

import os
import re
from pathlib import Path

# Mapeo de modelos antiguos a nuevos (v1beta 2026)
MODEL_MAP = {
    # LLM Models
    "gemini-1.5-flash": "gemini-2.0-flash-exp",
    "gemini-1.5-pro": "gemini-2.0-flash-thinking-exp",
    "gemini-1.0-pro": "gemini-2.0-flash-exp",
    
    # Embedding Models
    "models/embedding-001": "models/gemini-embedding-001",
    "text-embedding-001": "gemini-embedding-001",
}

def migrate_file(file_path: Path) -> bool:
    """Migrar un archivo .py específico."""
    if not file_path.exists():
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    changes_made = 0
    
    # Reemplazar modelos
    for old_model, new_model in MODEL_MAP.items():
        # Buscar en strings de Python (comillas simples y dobles)
        pattern = f'["\']?{re.escape(old_model)}["\']?'
        
        if old_model in content:
            content = re.sub(pattern, f'"{new_model}"', content)
            changes_made += 1
            print(f"  ✓ {old_model} → {new_model}")
    
    # Guardar si hubo cambios
    if changes_made > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  📝 Archivo actualizado: {file_path}")
        return True
    
    return False

def migrate_project():
    """Migrar todo el proyecto."""
    print("=" * 80)
    print("🔄 MIGRACIÓN DE MODELOS GEMINI (v1beta)")
    print("=" * 80)
    
    # Archivos a migrar
    files_to_migrate = [
        Path("proyectos/06-auditor-legal-crag-selfrag/src/config.py"),
        Path("proyectos/06-auditor-legal-crag-selfrag/src/agent/nodes.py"),
        Path("proyectos/06-auditor-legal-crag-selfrag/src/ingestion/processor.py"),
        Path(".env"),
    ]
    
    total_changes = 0
    
    for file_path in files_to_migrate:
        if file_path.exists():
            print(f"\n📄 Migrando: {file_path}")
            if migrate_file(file_path):
                total_changes += 1
        else:
            print(f"  ⚠️ No encontrado: {file_path}")
    
    print("\n" + "=" * 80)
    if total_changes > 0:
        print(f"✅ MIGRACIÓN COMPLETADA: {total_changes} archivo(s) actualizado(s)")
        print("\n⚠️ IMPORTANTE:")
        print("  1. Eliminar storage antiguo: Remove-Item -Recurse -Force storage\\")
        print("  2. Limpiar cache: Remove-Item -Recurse -Filter __pycache__")
        print("  3. Probar: python proyectos/06-auditor-legal-crag-selfrag/cli.py")
    else:
        print("⚠️ No se encontraron archivos para migrar")
    print("=" * 80)

if __name__ == "__main__":
    migrate_project()
