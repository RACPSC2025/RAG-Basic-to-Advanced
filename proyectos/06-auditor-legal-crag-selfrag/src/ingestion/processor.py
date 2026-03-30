import time
import os
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from llama_parse import LlamaParse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from ..config import Config
from ..utils.token_counter import TokenAuditor
from ..utils.visuals import IngestionUI, console

logger = logging.getLogger(__name__)

class LegalIngestor:
    """
    Motor de ingesta con Paciencia Extrema para Free Tier.
    """
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.auditor = TokenAuditor()
        self.embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
        if not dry_run:
            self.vector_store = Chroma(
                collection_name=Config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=Config.CHROMA_PATH
            )
        
        self.parser = LlamaParse(
            result_type="markdown",
            language="es",
            verbose=False,
            api_key=Config.LLAMA_PARSE_API_KEY
        )

    @retry(
        wait=wait_exponential(multiplier=2, min=10, max=120), # Más agresivo en la espera
        stop=stop_after_attempt(7),                          # Más intentos
        retry=retry_if_exception_type(Exception)
    )
    def _safe_add_documents(self, batch: List[Document]):
        """Añade documentos con reintentos y esperas largas."""
        if not self.dry_run:
            try:
                self.vector_store.add_documents(batch)
                # Pausa de seguridad larga para resetear el contador de la API
                time.sleep(12) 
            except Exception as e:
                logger.error(f"Error en _safe_add_documents: {str(e)}")
                raise e

    def process_files(self, file_paths: List[str], batch_size: int = 5):
        """Procesa archivos con lotes pequeños para evitar 429."""
        start_time = time.time()
        stats = {"parsing_speed": 0.0, "embedding_speed": 0.0}
        success = True
        
        with IngestionUI.create_progress() as progress:
            # 1. Parsing
            parse_task = progress.add_task("[magenta]Parsing PDFs...", total=len(file_paths))
            all_documents = []
            parsing_start = time.time()
            
            for path in file_paths:
                if self.dry_run:
                    time.sleep(0.5)
                    all_documents.append(Document(page_content=f"Simulación de {path}"))
                else:
                    llama_docs = self.parser.load_data(path)
                    for ldoc in llama_docs:
                        all_documents.append(Document(page_content=ldoc.text, metadata={"source": path}))
                progress.update(parse_task, advance=1)

            stats["parsing_speed"] = len(file_paths) / max((time.time() - parsing_start) / 60, 0.001)

            # 2. Indexación con lotes muy pequeños (5)
            index_task = progress.add_task("[green]Indexando (Lotes de 5)...", total=len(all_documents))
            indexing_start = time.time()
            
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                try:
                    if not self.dry_run:
                        self._safe_add_documents(batch)
                        for doc in batch:
                            self.auditor.add_usage(input_text=doc.page_content)
                    else:
                        time.sleep(0.5)
                    
                    progress.update(index_task, advance=len(batch))
                except Exception as e:
                    console.print(f"\n[bold red]❌ Error fatal en lote {i}: {str(e)}[/bold red]")
                    success = False
                    break

        if success:
            IngestionUI.show_final_stats(stats)
            IngestionUI.show_token_report(self.auditor.get_summary())
            console.print(f"\n[bold green]✅ Ingesta COMPLETADA al 100%[/bold green]")
        else:
            console.print(f"\n[bold yellow]⚠️ Ingesta PARCIAL: El proceso se detuvo para evitar un bloqueo mayor.[/bold yellow]")
