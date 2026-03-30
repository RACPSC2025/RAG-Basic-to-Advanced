"""
Auditor de Tokens - Proyecto 6

Migrado a AWS Bedrock - 2026-03-30
"""

import time
from typing import Dict


class TokenAuditor:
    """
    Auditor local para el seguimiento de cuotas de API.
    Actualizado para AWS Bedrock (Titan Embeddings).
    """
    
    def __init__(self, model_name: str = "amazon.titan-embed-text-v2:0"):
        self.model = model_name
        self.used_tokens = 0
        # Estimación para Titan Embeddings v2
        # 1 token ≈ 4 caracteres en promedio para español
        self.chars_per_token = 4
        
        # Rate limits (ajustables según tu cuota de AWS)
        self.tpm_limit = 1_000_000  # Tokens Per Minute (estimado)
        self.rpm_limit = 50         # Requests Per Minute (estimado)
        self.daily_limit = 10_000   # Requests Per Day (estimado)

        self.start_time = time.time()
        self.request_count = 0

    def add_usage(self, input_text: str, output_text: str = ""):
        """Estima y acumula el uso de tokens."""
        # Estimación: 1 token cada 4 caracteres para español
        input_tokens = len(input_text) // self.chars_per_token
        output_tokens = len(output_text) // self.chars_per_token if output_text else 0
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
