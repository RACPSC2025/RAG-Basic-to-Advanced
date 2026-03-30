import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.processor import LegalIngestor
from src.agent.graph import create_auditor_graph
from src.utils.visuals import console, IngestionUI

class AuditorCLI:
    def __init__(self):
        # El grafo se inicializa solo una vez para ahorrar recursos
        self.graph = create_auditor_graph()
        self.ingestor = None

    def show_menu(self):
        table = Table(title="⚖️ SISTEMA DE AUDITORÍA LEGAL COLOMBIANA", title_style="bold cyan")
        table.add_column("Opción", style="magenta")
        table.add_column("Descripción", style="white")
        
        table.add_row("1", "📥 Ingestar nuevos documentos (PDF)")
        table.add_row("2", "💬 Chatear con el Auditor (CRAG + Self-RAG)")
        table.add_row("3", "📊 Ver configuración del sistema")
        table.add_row("4", "❌ Salir")
        
        console.print("\n")
        console.print(Panel(table, border_style="cyan", padding=(1, 2)))

    def run_ingestion(self):
        path = Prompt.ask("[bold magenta]Introduce la ruta del archivo o carpeta PDF[/]", default="data/")
        dry_run = Prompt.ask("¿Ejecutar en modo simulacro (dry-run)?", choices=["s", "n"], default="n") == "s"
        
        if not self.ingestor:
            self.ingestor = LegalIngestor(dry_run=dry_run)
        
        import glob
        if os.path.isfile(path):
            files = [path]
        else:
            files = glob.glob(os.path.join(path, "*.pdf"))
            
        if not files:
            console.print("[bold red]❌ No se encontraron archivos PDF en la ruta especificada.[/]")
            return
            
        self.ingestor.process_files(files)

    def run_chat(self):
        console.print(Panel("[bold green]⚖️ MODO CHAT ACTIVADO[/]\n[italic]Escriba su consulta legal colombiana. Escriba 'salir' para volver al menú principal.[/]", border_style="green"))
        
        while True:
            question = Prompt.ask("\n[bold cyan]Abogado, ¿cuál es su consulta legal?[/]")
            
            if question.lower() in ["salir", "exit", "quit", "4"]:
                break
                
            inputs = {
                "question": question,
                "retries": 0,
                "steps": [],
                "documents": []
            }
            
            console.print("\n[italic yellow]Investigando en la base de datos jurídica...[/]")
            
            # Ejecutar el grafo
            final_output = None
            for output in self.graph.stream(inputs):
                for key, value in output.items():
                    # Iconografía profesional por fase
                    icon = "🔍" if "retrieve" in key else "⚖️" if "grade" in key else "✍️" if "generate" in key else "🔄"
                    console.print(f"[bold blue]{icon} [Step]: {key.upper()}[/]")
                    final_output = value # Capturamos el último estado para la respuesta final
            
            if final_output and "generation" in final_output:
                console.print("\n" + "="*50)
                console.print(Panel(final_output["generation"], title="[bold green]Respuesta del Auditor Legal[/]", border_style="green", expand=True))
                console.print("="*50 + "\n")

    def show_config(self):
        from src.config import Config
        table = Table(title="⚙️ PARÁMETROS DEL SISTEMA")
        table.add_column("Configuración", style="cyan")
        table.add_column("Valor Actual", style="magenta")
        
        table.add_row("Modelo de Lenguaje (LLM)", Config.LLM_MODEL)
        table.add_row("Modelo de Embeddings", Config.EMBEDDING_MODEL)
        table.add_row("Límite de Peticiones (RPM)", str(Config.RPM))
        table.add_row("Pausa entre Pasos", f"{Config.REQUEST_DELAY:.2f}s")
        table.add_row("Límite de Reintentos", str(Config.MAX_RETRIES))
        
        console.print(Panel(table, border_style="blue", padding=(1, 2)))

    def main_loop(self):
        while True:
            self.show_menu()
            # CORRECCIÓN: choices debe ser lista de strings
            choice = Prompt.ask(
                "[bold yellow]Seleccione una opción[/]", 
                choices=["1", "2", "3", "4"], 
                default="2"
            )
            
            if choice == "1":
                self.run_ingestion()
            elif choice == "2":
                self.run_chat()
            elif choice == "3":
                self.show_config()
            elif choice == "4":
                console.print("\n[bold cyan]🏛️ Cerrando el despacho legal. ¡Éxito en sus casos![/]\n")
                break

if __name__ == "__main__":
    cli = AuditorCLI()
    cli.main_loop()
