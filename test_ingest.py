"""
Test directo de ingesta revisado — Ruta Definitiva
"""
import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

sys.path.append(os.path.dirname(os.path.abspath(__name__)))

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from proyectos.Rag_Legal.ingestor import ingest_pdf

pdf_path = os.path.join("data", "input", "sample.pdf")
print(f"CWD: {os.getcwd()}")
print(f"Buscando: {os.path.abspath(pdf_path)}")

if not os.path.exists(pdf_path):
    print(f"❌ No se encontró: {pdf_path}")
    sys.exit(1)

print(f"Iniciando ingesta de: {pdf_path}...")
try:
    stats = ingest_pdf(pdf_path)
    print(f"\n✅ EXITOSO: {stats}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ FALLÓ: {type(e).__name__}: {e}")
