"""
Ingestor de PDFs legales — Proyecto Fénix.

Técnicas implementadas:
  - Extracción con PyMuPDF (fitz) para preservar estructura de artículos
  - RecursiveCharacterTextSplitter adaptado a estructura legal colombiana
  - Batching con backoff exponencial (AWS Bedrock Titan Embed v2)
  - Deduplicación por hash antes de agregar a ChromaDB
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from proyectos.Rag_Legal.config import get_embeddings, settings

logger = logging.getLogger(__name__)


# ── Separadores legales colombianos (por orden de prioridad) ──────────────────
LEGAL_SEPARATORS = [
    "\nARTÍCULO ",
    "\nArtículo ",
    "\nCAPÍTULO ",
    "\nSECCIÓN ",
    "\nPARÁGRAFO",
    "\n\n",
    "\n",
    " ",
]


def _extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extrae texto preservando saltos de línea y estructura del documento.
    Usa PyMuPDF (fitz) que respeta mejor la maquetación de decretos colombianos.
    """
    doc = fitz.open(str(pdf_path))
    pages: List[str] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append(f"\n--- Página {page_num} ---\n{text}")
    doc.close()
    return "\n".join(pages)


def _split_document(text: str, source: str) -> List[Document]:
    """
    Divide el texto usando separadores adaptados a la estructura normativa colombiana.
    Añade metadata de origen en cada chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": Path(source).name, "path": str(source)}],
    )
    logger.info(f"Documento '{Path(source).name}' → {len(chunks)} chunks generados.")
    return chunks


def _compute_hash(text: str) -> str:
    """Hash MD5 del contenido del chunk para deduplicación."""
    return hashlib.md5(text.encode()).hexdigest()


def _batch_embed_with_backoff(
    vector_store: Chroma,
    docs: List[Document],
    batch_size: int,
    max_retries: int,
) -> int:
    """
    Inserta documentos en Chroma en batches con backoff exponencial.
    Respeta el rate limit de 10 RPM de la Free API Key de Gemini.
    """
    total_added = 0
    ids_in_store = set(vector_store.get()["ids"])

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]

        # Deduplicación
        new_docs, new_ids = [], []
        for doc in batch:
            doc_id = _compute_hash(doc.page_content)
            if doc_id not in ids_in_store:
                doc.metadata["chunk_id"] = doc_id
                new_docs.append(doc)
                new_ids.append(doc_id)

        if not new_docs:
            logger.debug(f"Batch {i // batch_size + 1}: todos los chunks ya existen, saltando.")
            continue

        # Inserción con reintentos
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=new_docs, ids=new_ids)
                ids_in_store.update(new_ids)
                total_added += len(new_docs)
                logger.info(
                    f"Batch {i // batch_size + 1}: +{len(new_docs)} chunks insertados."
                )
                # Pausa para respetar RPM de AWS Bedrock (configurado en .env)
                pause = 60 / settings.RPM
                logger.debug(f"Pausa de {pause:.1f}s entre batches (RPM={settings.RPM}).")
                time.sleep(pause)
                break
            except Exception as e:
                wait = 2 ** attempt * 5
                logger.warning(
                    f"Error en batch {i // batch_size + 1} (intento {attempt + 1}): {e}. "
                    f"Reintentando en {wait}s..."
                )
                time.sleep(wait)
        else:
            logger.error(f"Batch {i // batch_size + 1} falló tras {max_retries} intentos.")

    return total_added


def ingest_pdf(pdf_path: str | Path) -> dict:
    """
    Pipeline completo de ingesta:
      1. Extrae texto del PDF con PyMuPDF
      2. Divide en chunks con separadores legales colombianos
      3. Vectoriza en batches con backoff exponencial
      4. Persiste en ChromaDB existente en /storage

    Returns:
        dict con estadísticas de la ingesta
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    logger.info(f"Iniciando ingesta: {pdf_path.name}")

    # 1. Extracción
    text = _extract_text_from_pdf(pdf_path)
    logger.info(f"Texto extraído: {len(text):,} caracteres.")

    # 2. Chunking
    docs = _split_document(text, str(pdf_path))

    # 3. Conexion al store con AWS Bedrock Titan Embeddings v2
    logger.info("Inicializando AWS Bedrock embeddings (Titan Embed v2)...")
    embeddings = get_embeddings()
    logger.info(f"Conectando a ChromaDB en: {settings.STORAGE_PATH} | Colección: {settings.COLLECTION_NAME}")
    vector_store = Chroma(
        persist_directory=settings.STORAGE_PATH,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
    )
    logger.info("Conexion a ChromaDB exitosa.")

    # 4. Inserción por batches
    added = _batch_embed_with_backoff(
        vector_store=vector_store,
        docs=docs,
        batch_size=settings.BATCH_SIZE,
        max_retries=settings.MAX_RETRIES,
    )

    return {
        "file": pdf_path.name,
        "total_chunks": len(docs),
        "new_chunks_added": added,
        "duplicates_skipped": len(docs) - added,
        "storage_path": settings.STORAGE_PATH,
    }


def get_vector_store() -> Chroma:
    """
    Acceso de solo lectura al vector store existente.
    Usado internamente por el retriever del grafo.
    """
    return Chroma(
        persist_directory=settings.STORAGE_PATH,
        embedding_function=get_embeddings(),
        collection_name=settings.COLLECTION_NAME,
    )
