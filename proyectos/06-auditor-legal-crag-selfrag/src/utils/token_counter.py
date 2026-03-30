import time
from typing import Dict

class TokenAuditor:
    """
    Auditor local para el seguimiento de cuotas de la API de Gemini.
    Basado en límites de Free Tier (estimados para 2026).
    """
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model = model_name
        self.used_tokens = 0
        # Límites teóricos Free Tier (ajustables)
        self.tpm_limit = 1_000_000  # Tokens Per Minute
        self.rpm_limit = 15         # Requests Per Minute
        self.daily_limit = 1_500    # Requests Per Day (aproximado)
        
        self.start_time = time.time()
        self.request_count = 0

    def add_usage(self, input_text: str, output_text: str = ""):
        """Estima y acumula el uso de tokens."""
        # Estimación conservadora: 1 token cada 3.5 caracteres para español
        input_tokens = len(input_text) // 3.5
        output_tokens = len(output_text) // 3.5
        total = int(input_tokens + output_tokens)
        
        self.used_tokens += total
        self.request_count += 1
        return total

    def get_summary(self) -> Dict:
        """Retorna un resumen del consumo actual."""
        elapsed = (time.time() - self.start_time) / 60
        return {
            "total_used": self.used_tokens,
            "tpm_avg": int(self.used_tokens / max(elapsed, 1)),
            "requests": self.request_count,
            "remaining_estimate": max(0, self.tpm_limit - self.used_tokens)
        }
