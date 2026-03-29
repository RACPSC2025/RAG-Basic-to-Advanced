# Módulo 13: Producción y Deploy

> **Basado en**: [Documentación Oficial de LangGraph Deploy](https://docs.langchain.com/oss/python/langgraph/deploy) y [LangSmith Observability](https://docs.langchain.com/oss/python/langgraph/observability)  
> **Estado**: ✅ COMPLETADO - Listo para Producción  
> **Prerrequisitos**: Módulos 1-12 completados

---

## 📋 Índice del Módulo

### Parte 1: Preparación para Producción
1. [13.1 - Estructura de Aplicación](#131-estructura-de-aplicación)
2. [13.2 - Testing y Validación](#132-testing-y-validación)
3. [13.3 - Variables de Entorno y Seguridad](#133-variables-de-entorno-y-seguridad)

### Parte 2: Deploy
4. [13.4 - Dockerización](#134-dockerización)
5. [13.5 - Deploy con LangSmith Cloud](#135-deploy-con-langsmith-cloud)
6. [13.6 - Deploy con Kubernetes](#136-deploy-con-kubernetes)

### Parte 3: Monitoreo y Optimización
7. [13.7 - LangSmith Observability](#137-langsmith-observability)
8. [13.8 - Optimización de Costos](#138-optimización-de-costos)
9. [13.9 - Optimización de Performance](#139-optimización-de-performance)

### Parte 4: CI/CD y Mantenimiento
10. [13.10 - CI/CD Pipeline](#1310-cicd-pipeline)
11. [13.11 - Mantenimiento y Actualizaciones](#1311-mantenimiento-y-actualizaciones)

---

## Parte 1: Preparación para Producción

### 13.1 - Estructura de Aplicación

#### Estructura Recomendada

```
RAG MVP/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── legal_assistant.py      # Agente principal
│   │   ├── classifier.py           # Clasificador
│   │   └── tools.py                # Herramientas
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py              # Configuración de logging
│   │   └── config.py               # Configuración centralizada
│   └── api/
│       ├── __init__.py
│       ├── routes.py               # Endpoints API
│       └── schemas.py              # Pydantic schemas
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py              # Tests de agentes
│   ├── test_tools.py               # Tests de herramientas
│   └── test_integration.py         # Tests de integración
│
├── langgraph.json                  # Configuración LangGraph
├── Dockerfile                      # Dockerfile para producción
├── docker-compose.yml              # Orquestación de contenedores
├── requirements.txt                # Dependencias
├── .env.example                    # Ejemplo de variables de entorno
├── .gitignore                      # Ignorar archivos sensibles
└── README.md                       # Documentación
```

#### langgraph.json

```json
{
  "name": "legal-rag-agent",
  "version": "1.0.0",
  "description": "Agente RAG Legal para producción",
  "agents": {
    "legal_assistant": {
      "file": "./src/agents/legal_assistant.py",
      "graph": "graph"
    },
    "classifier": {
      "file": "./src/agents/classifier.py",
      "graph": "graph"
    }
  },
  "env": {
    "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
    "QDRANT_URL": "${QDRANT_URL}",
    "LANGSMITH_TRACING": "true"
  },
  "python_version": "3.12"
}
```

---

### 13.2 - Testing y Validación

#### Tests Unitarios

```python
# tests/test_agents.py
import pytest
from src.agents.legal_assistant import AgenticRAG, AgentConfig

class TestAgenticRAG:
    """Tests para el agente RAG."""
    
    def test_initialization(self):
        """Prueba de inicialización."""
        config = AgentConfig()
        agent = AgenticRAG(config)
        
        assert agent is not None
        assert agent.graph is not None
    
    def test_invoke_basic(self):
        """Prueba de invocación básica."""
        config = AgentConfig()
        agent = AgenticRAG(config)
        
        result = agent.invoke("¿Qué es una tutela?")
        
        assert result is not None
        assert "answer" in result
        assert len(result["answer"]) > 0
    
    def test_classification_accuracy(self):
        """Prueba de precisión de clasificación."""
        config = AgentConfig()
        agent = AgenticRAG(config)
        
        # Consultas de prueba
        test_cases = [
            ("¿Me pueden despedir?", "laboral"),
            ("¿Cómo creo una SAS?", "corporativo"),
            ("¿Qué pena tiene el hurto?", "penal")
        ]
        
        for query, expected_area in test_cases:
            result = agent.invoke(query)
            # Verificar que el agente clasifica correctamente
            assert result["clasificacion"].area == expected_area
```

#### Tests de Integración

```python
# tests/test_integration.py
import pytest
from src.api.routes import app

class TestAPIIntegration:
    """Tests de integración de API."""
    
    @pytest.fixture
    def client(self):
        """Cliente de prueba."""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_consult_endpoint(self, client):
        """Prueba de endpoint de consulta."""
        response = client.post(
            "/api/consultar",
            json={
                "pregunta": "¿Qué es una tutela?",
                "empleado_id": "TEST001"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "respuesta" in data
        assert "area" in data
    
    def test_rate_limiting(self, client):
        """Prueba de rate limiting."""
        # Hacer 100 requests rápidos
        responses = []
        for i in range(100):
            response = client.post(
                "/api/consultar",
                json={
                    "pregunta": f"Consulta {i}",
                    "empleado_id": "TEST001"
                }
            )
            responses.append(response.status_code)
        
        # Algunos deberían ser 429 (Too Many Requests)
        assert 429 in responses
```

---

### 13.3 - Variables de Entorno y Seguridad

#### .env.example

```bash
# ===========================================
# CONFIGURACIÓN DE PRODUCCIÓN
# ===========================================

# ===========================================
# Google Gemini API
# ===========================================
GOOGLE_API_KEY=tu_api_key_aqui
GOOGLE_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta

# ===========================================
# Qdrant Vector Store
# ===========================================
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documentos_legales

# ===========================================
# LangSmith (Observability)
# ===========================================
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=tu_langsmith_api_key
LANGSMITH_PROJECT=legal-rag-production
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# ===========================================
# Base de Datos (Persistencia)
# ===========================================
DATABASE_URL=postgresql://user:password@localhost:5432/legal_rag
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=legal_rag

# ===========================================
# Configuración de la Aplicación
# ===========================================
APP_NAME=Legal RAG Agent
APP_VERSION=1.0.0
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=60

# ===========================================
# Rate Limiting
# ===========================================
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# ===========================================
# Seguridad
# ===========================================
SECRET_KEY=tu_secret_key_muy_largo_y_seguro
ALLOWED_HOSTS=["localhost", "127.0.0.1", "tudominio.com"]
CORS_ORIGINS=["https://tudominio.com"]

# ===========================================
# Monitoreo
# ===========================================
ENABLE_METRICS=true
METRICS_PORT=9090
```

#### Seguridad en Producción

```python
# src/utils/security.py
from functools import wraps
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

security = HTTPBearer()

def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Verificar API Key en requests.
    
    Args:
        credentials: Credenciales HTTP
    
    Returns:
        API Key válida
    
    Raises:
        HTTPException: Si la API Key es inválida
    """
    
    expected_key = os.getenv("API_KEY")
    
    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="API_KEY no configurada en servidor"
        )
    
    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida"
        )
    
    return credentials.credentials

# Uso en endpoints
@app.post("/api/consultar")
async def consultar(
    request: ConsultaRequest,
    api_key: str = Depends(verify_api_key)
):
    # Lógica del endpoint
    ...
```

---

## Parte 2: Deploy

### 13.4 - Dockerización

#### Dockerfile de Producción

```dockerfile
# ===========================================
# Stage 1: Build
# ===========================================
FROM python:3.12-slim as builder

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
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

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  # ===========================================
  # Aplicación Principal
  # ===========================================
  legal-rag:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - LANGSMITH_TRACING=true
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - DATABASE_URL=postgresql://user:password@postgres:5432/legal_rag
    depends_on:
      qdrant:
        condition: service_healthy
      postgres:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    networks:
      - legal-rag-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  # ===========================================
  # Qdrant Vector Store
  # ===========================================
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - legal-rag-network
    restart: unless-stopped

  # ===========================================
  # PostgreSQL (Persistencia)
  # ===========================================
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=legal_rag
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d legal_rag"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - legal-rag-network
    restart: unless-stopped

  # ===========================================
  # Prometheus (Métricas)
  # ===========================================
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - legal-rag-network
    restart: unless-stopped

  # ===========================================
  # Grafana (Dashboards)
  # ===========================================
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    networks:
      - legal-rag-network
    restart: unless-stopped

volumes:
  qdrant_storage:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  legal-rag-network:
    driver: bridge
```

---

### 13.5 - Deploy con LangSmith Cloud

#### Pasos para Deploy

**Paso 1: Crear Repositorio GitHub**

```bash
# Inicializar repositorio
git init
git add .
git commit -m "Initial commit: Legal RAG Agent"

# Conectar con GitHub
git remote add origin https://github.com/tu-usuario/legal-rag.git
git push -u origin main
```

**Paso 2: Configurar LangSmith**

1. Ir a [LangSmith](https://smith.langchain.com/)
2. Iniciar sesión
3. Navegar a **Deployments**
4. Click en **+ New Deployment**
5. Conectar cuenta de GitHub
6. Seleccionar repositorio
7. Click en **Submit**

**Paso 3: Verificar Deploy**

El deploy toma ~15 minutos. Verificar estado en:
- **Deployment details** → Estado del deploy
- **Studio** → Visualizar grafo
- **API URL** → Obtener endpoint

**Paso 4: Probar API**

```python
from langgraph_sdk import get_sync_client

# Configurar cliente
client = get_sync_client(
    url="https://tu-deployment.langchain.com",
    api_key="tu_langsmith_api_key"
)

# Probar agente
for chunk in client.runs.stream(
    None,
    "legal_assistant",
    input={
        "messages": [{
            "role": "human",
            "content": "¿Qué es una tutela?"
        }]
    },
    stream_mode="updates"
):
    print(f"Event: {chunk.event}")
    print(f"Data: {chunk.data}")
```

---

### 13.6 - Deploy con Kubernetes

#### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legal-rag-agent
  labels:
    app: legal-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: legal-rag
  template:
    metadata:
      labels:
        app: legal-rag
    spec:
      containers:
      - name: legal-rag
        image: tu-registry/legal-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: google-api-key
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: LANGSMITH_TRACING
          value: "true"
        - name: LANGSMITH_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: langsmith-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      imagePullSecrets:
      - name: registry-secret

---
apiVersion: v1
kind: Service
metadata:
  name: legal-rag-service
spec:
  selector:
    app: legal-rag
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
stringData:
  google-api-key: "tu_google_api_key"
  langsmith-api-key: "tu_langsmith_api_key"
```

#### Deploy a Kubernetes

```bash
# Crear secretos
kubectl create secret generic api-secrets \
  --from-literal=google-api-key=tu_google_api_key \
  --from-literal=langsmith-api-key=tu_langsmith_api_key

# Aplicar deployment
kubectl apply -f k8s/deployment.yaml

# Verificar estado
kubectl get deployments
kubectl get pods
kubectl get services

# Ver logs
kubectl logs -f deployment/legal-rag-agent

# Escalar
kubectl scale deployment legal-rag-agent --replicas=5
```

---

## Parte 3: Monitoreo y Optimización

### 13.7 - LangSmith Observability

#### Habilitar Tracing

```python
# Configurar variables de entorno
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "tu_api_key"
os.environ["LANGSMITH_PROJECT"] = "legal-rag-production"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
```

#### Trazas Selectivas

```python
import langsmith as ls

# Traza SOLO esta invocación
with ls.tracing_context(enabled=True, project_name="debug-session"):
    result = agent.invoke({"messages": [{"role": "user", "content": "Query"}]})

# No trazar esta invocación
with ls.tracing_context(enabled=False):
    result = agent.invoke({"messages": [{"role": "user", "content": "Query"}]})
```

#### Metadata Personalizada

```python
# Agregar metadata y tags a trazas
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Query"}]},
    config={
        "tags": ["production", "legal-assistant", "v1.0"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production",
            "query_type": "laboral"
        }
    }
)
```

#### Anonimizar Datos Sensibles

```python
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

# Crear anonymizer para SSN
anonymizer = create_anonymizer([
    {"pattern": r"\b\d{3}-?\d{2}-?\d{4}\b", "replace": "<ssn>"},
    {"pattern": r"\b\d{10,}\b", "replace": "<account_number>"}
])

# Aplicar al tracer
tracer_client = Client(anonymizer=anonymizer)
tracer = LangChainTracer(client=tracer_client)

# Configurar grafo con tracer
graph = graph.compile().with_config({'callbacks': [tracer]})
```

---

### 13.8 - Optimización de Costos

#### Estrategias de Optimización

```python
# 1. Cache de respuestas
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embedding(text_hash: str):
    """Cache de embeddings."""
    return get_embedding(text_hash)

def get_cached_embedding(text: str):
    """Obtener embedding con cache."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return cached_embedding(text_hash)

# 2. Batch de requests
async def batch_embeddings(texts: list[str], batch_size: int = 32):
    """Procesar embeddings en batches."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_result = await get_embeddings_batch(batch)
        results.extend(batch_result)
    return results

# 3. Limitar tokens de salida
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    max_output_tokens=500,  # Limitar salida
    temperature=0.3
)

# 4. Usar modelo más económico para tareas simples
llm_simple = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # Más barato
llm_complex = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")  # Más caro

# Router de modelos
def get_llm_for_task(task_complexity: str):
    if task_complexity == "simple":
        return llm_simple
    else:
        return llm_complex
```

#### Monitoreo de Costos

```python
# src/utils/cost_tracking.py
from langchain.callbacks import get_openai_callback

def track_costs(func):
    """Decorator para trackear costos."""
    def wrapper(*args, **kwargs):
        with get_openai_callback() as cb:
            result = func(*args, **kwargs)
            
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost: ${cb.total_cost:.4f}")
            
            return result
    return wrapper

@track_costs
def process_query(query: str):
    """Procesar query con tracking de costos."""
    # Lógica de procesamiento
    ...
```

---

### 13.9 - Optimización de Performance

#### Técnicas de Optimización

```python
# 1. Connection Pooling para Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333,
    connection_pool_size=10,  # Pool de conexiones
    timeout=30
)

# 2. Async para I/O operations
import asyncio
from aiohttp import ClientSession

async def fetch_multiple_sources(urls: list[str]):
    """Fetch múltiple URLs en paralelo."""
    async with ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

# 3. Streaming para respuestas largas
async def stream_response(query: str):
    """Stream de respuesta en lugar de wait completo."""
    for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}):
        yield chunk

# 4. Parallelización de retrieval
from concurrent.futures import ThreadPoolExecutor

def parallel_retrieval(queries: list[str], max_workers: int = 5):
    """Búsqueda en paralelo."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(retrieve_single, queries))
    return results
```

#### Métricas de Performance

```python
# src/utils/metrics.py
import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

# Métricas
REQUEST_COUNT = Counter('agent_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('agent_request_latency_seconds', 'Request latency')
ACTIVE_REQUESTS = Gauge('agent_active_requests', 'Active requests')
TOKEN_COUNT = Counter('agent_tokens_total', 'Total tokens', ['type'])

def track_metrics(func):
    """Decorator para métricas."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        ACTIVE_REQUESTS.inc()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.observe(latency)
            ACTIVE_REQUESTS.dec()
    
    return wrapper
```

---

## Parte 4: CI/CD y Mantenimiento

### 13.10 - CI/CD Pipeline

#### GitHub Actions

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # ===========================================
  # Test
  # ===========================================
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  # ===========================================
  # Lint
  # ===========================================
  lint:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install linters
      run: |
        pip install flake8 black mypy
    
    - name: Run flake8
      run: flake8 src/
    
    - name: Run black
      run: black --check src/
    
    - name: Run mypy
      run: mypy src/

  # ===========================================
  # Build Docker
  # ===========================================
  build:
    runs-on: ubuntu-latest
    needs: [test, lint]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ghcr.io/${{ github.repository }}:${{ github.sha }}

  # ===========================================
  # Deploy to LangSmith
  # ===========================================
  deploy:
    runs-on: ubuntu-latest
    needs: [build]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to LangSmith Cloud
      env:
        LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
      run: |
        pip install langgraph-cli
        langgraph deploy --api-key $LANGSMITH_API_KEY
```

---

### 13.11 - Mantenimiento y Actualizaciones

#### Checklist de Mantenimiento

```markdown
# Checklist de Mantenimiento Semanal

## Monitoreo
- [ ] Revisar dashboards de LangSmith
- [ ] Verificar métricas de performance
- [ ] Revisar logs de errores
- [ ] Monitorear costos de API

## Actualizaciones
- [ ] Actualizar dependencias de seguridad
- [ ] Revisar changelog de LangChain/LangGraph
- [ ] Actualizar modelos de LLM si hay mejoras

## Backup
- [ ] Verificar backups de vector store
- [ ] Verificar backups de base de datos
- [ ] Probar restore de backups

## Documentación
- [ ] Actualizar documentación de API
- [ ] Documentar cambios importantes
- [ ] Actualizar runbooks de operaciones
```

#### Rollback en Caso de Errores

```bash
# Rollback de deployment en Kubernetes
kubectl rollout undo deployment/legal-rag-agent

# Verificar estado del rollback
kubectl rollout status deployment/legal-rag-agent

# Rollback a versión específica
kubectl rollout undo deployment/legal-rag-agent --to-revision=2
```

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [LangGraph Deploy](https://docs.langchain.com/oss/python/langgraph/deploy)
- [LangSmith Observability](https://docs.langchain.com/oss/python/langgraph/observability)
- [LangSmith Cloud](https://docs.langchain.com/langsmith/deploy-to-cloud)

### Herramientas Recomendadas
- **Docker**: [docker.com](https://www.docker.com/)
- **Kubernetes**: [kubernetes.io](https://kubernetes.io/)
- **Prometheus**: [prometheus.io](https://prometheus.io/)
- **Grafana**: [grafana.com](https://grafana.com/)

---

*Siguiente: Módulo 14 - Proyecto Final*
