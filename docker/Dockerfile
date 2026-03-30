# ===========================================
# Dockerfile de Producción - Legal RAG Agent
# ===========================================
# Versión: 1.0.0
# Basado en: Python 3.12-slim
# ===========================================

# ===========================================
# Stage 1: Build
# ===========================================
FROM python:3.12-slim as builder

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --user -r requirements.txt

# ===========================================
# Stage 2: Runtime
# ===========================================
FROM python:3.12-slim

WORKDIR /app

# Crear usuario no-root por seguridad
RUN useradd -m -u 1000 appuser

# Copiar dependencias del builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copiar código de la aplicación
COPY src/ ./src/
COPY langgraph.json .

# Configurar PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app/src

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
