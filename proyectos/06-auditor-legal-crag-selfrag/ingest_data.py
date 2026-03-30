import sys
import os
import argparse
from glob import glob

# Añadir el directorio actual al path para importaciones relativas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.processor import LegalIngestor
from src.utils.visuals import console

def main():
    parser = argparse.ArgumentParser(description="🚀 Ingestor Legal Profesional (CRAG + Self-RAG)")
    parser.add_argument("--path", type=str, default="data/sample.pdf", help="Ruta al archivo o carpeta PDF")
    parser.add_argument("--dry-run", action="store_true", help="Simula el proceso sin usar tokens reales")
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]--- ⚖️ INICIANDO PROCESO DE INGESTA LEGAL ---[/bold cyan]")
    
    # 1. Buscar archivos
    if os.path.isfile(args.path):
        files = [args.path]
    else:
        # Buscar todos los PDFs en la ruta indicada
        files = glob(os.path.join(args.path, "*.pdf"))
    
    if not files:
        console.print(f"[bold red]❌ Error: No se encontraron archivos PDF en {args.path}[/bold red]")
        sys.exit(1)

    console.print(f"📦 Archivos encontrados: {len(files)}")
    if args.dry_run:
        console.print("[yellow]⚠️ MODO SIMULACRO ACTIVADO (Sin coste de tokens)[/yellow]")
    
    # 2. Iniciar Ingestor
    ingestor = LegalIngestor(dry_run=args.dry_run)
    
    # 3. Procesar
    try:
        ingestor.process_files(files)
    except Exception as e:
        console.print(f"[bold red]❌ Error crítico durante la ingesta: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
