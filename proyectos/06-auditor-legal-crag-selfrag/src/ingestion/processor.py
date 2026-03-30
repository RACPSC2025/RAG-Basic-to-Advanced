import time
import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from llama_parse import LlamaParse

from ..config import Config
from ..utils.token_counter import TokenAuditor
from ..utils.visuals import IngestionUI, console

class LegalIngestor:
    """
    Motor de ingesta de grado profesional para documentos legales colombianos.
    Integra LlamaParse, ChromaDB y monitoreo en tiempo real.
    """
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.auditor = TokenAuditor()
        self.embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
        # Configurar Vector Store (ChromaDB)
        if not dry_run:
            self.vector_store = Chroma(
                collection_name=Config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=Config.CHROMA_PATH
            )
        
        # Configurar LlamaParse
        self.parser = LlamaParse(
            result_type="markdown",
            language="es",
            verbose=False,
            api_key=Config.LLAMA_PARSE_API_KEY
        )

    def process_files(self, file_paths: List[str]):
        """Procesa una lista de archivos legales."""
        start_time = time.time()
        stats = {
            "parsing_speed": 0.0,
            "embedding_speed": 0.0,
            "io_latency": 0.0
        }
        
        with IngestionUI.create_progress() as progress:
            # 1. Tarea de Parsing
            parse_task = progress.add_task("[magenta]Parsing PDFs (LlamaParse)...", total=len(file_paths))
            
            all_documents = []
            parsing_start = time.time()
            
            for path in file_paths:
                if self.dry_run:
                    time.sleep(0.5) # Simulación
                    all_documents.append(Document(page_content=f"Simulación de {path}", metadata={"source": path}))
                else:
                    # LlamaParse preserva estructura legal (tablas, artículos)
                    llama_docs = self.parser.load_data(path)
                    for ldoc in llama_docs:
                        all_documents.append(Document(page_content=ldoc.text, metadata={"source": path}))
                
                progress.update(parse_task, advance=1)
                self.auditor.add_usage(input_text=path) # Registro de actividad

            elapsed_parsing = (time.time() - parsing_start) / 60
            stats["parsing_speed"] = len(file_paths) / max(elapsed_parsing, 0.001)

            # 2. Tarea de Indexación
            index_task = progress.add_task("[green]Generando Embeddings & Indexando...", total=len(all_documents))
            
            indexing_start = time.time()
            io_total_time = 0
            
            for doc in all_documents:
                chunk_start = time.time()
                
                if not self.dry_run:
                    # Guardar en ChromaDB
                    io_start = time.time()
                    self.vector_store.add_documents([doc])
                    io_total_time += (time.time() - io_start)
                    
                    # Registrar uso de tokens para el embedding
                    self.auditor.add_usage(input_text=doc.page_content)
                else:
                    time.sleep(0.1) # Simulación de latencia

                progress.update(index_task, advance=1)
            
            elapsed_indexing = time.time() - indexing_start
            stats["embedding_speed"] = len(all_documents) / max(elapsed_indexing, 0.001)
            stats["io_latency"] = io_total_time / max(len(all_documents), 1)

        # Mostrar reportes finales
        IngestionUI.show_final_stats(stats)
        IngestionUI.show_token_report(self.auditor.get_summary())
        
        console.print(f"\n[bold green]✅ Ingesta finalizada correctamente ({'Modo Simulacro' if self.dry_run else 'Modo Real'})[/bold green]")
