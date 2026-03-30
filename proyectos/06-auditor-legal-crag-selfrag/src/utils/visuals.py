from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    DownloadColumn,
    TransferSpeedColumn
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Dict

console = Console()

class IngestionUI:
    """Consola visual profesional para el proceso de ingesta RAG."""
    
    @staticmethod
    def create_progress():
        """Crea un gestor de progreso con métricas de rendimiento."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TransferSpeedColumn(), # Mide la velocidad de transferencia (docs/s o chunks/s)
            TimeRemainingColumn(),
            console=console
        )

    @staticmethod
    def show_token_report(summary: Dict):
        """Muestra el reporte de tokens en una tabla elegante."""
        table = Table(title="📊 REPORTE DE CONSUMO - GEMINI API", title_style="bold green")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="magenta")
        
        table.add_row("Tokens Utilizados (Est.)", f"{summary['total_used']:,}")
        table.add_row("Promedio Tokens/min", f"{summary['tpm_avg']:,}")
        table.add_row("Peticiones Realizadas", str(summary['requests']))
        table.add_row("Capacidad Restante TPM", f"{summary['remaining_estimate']:,}")
        
        console.print("\n")
        console.print(Panel(table, border_style="green", padding=(1, 2)))

    @staticmethod
    def show_final_stats(stats: Dict):
        """Muestra estadísticas finales de rendimiento."""
        table = Table(title="📈 ESTADÍSTICAS DE RENDIMIENTO", title_style="bold blue")
        table.add_column("Proceso", style="cyan")
        table.add_column("Métrica", style="magenta")
        
        table.add_row("Parsing Speed", f"{stats['parsing_speed']:.2f} docs/min")
        table.add_row("Embedding Throughput", f"{stats['embedding_speed']:.2f} chunks/s")
        table.add_row("IO Indexing Latency", f"{stats['io_latency']:.4f} s/chunk")
        
        console.print(Panel(table, border_style="blue", padding=(1, 2)))
