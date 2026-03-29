# 🏢 Casos Reales de Producción

> **Propósito**: Implementaciones reales de agentes para el sector legal  
> **Estado**: ✅ 2 casos implementados, más en desarrollo  
> **Nivel**: Producción (listo para adaptar)

---

## 📁 Casos Disponibles

### 1. Asistente Legal Corporativo

**Archivo**: `01_asistente_legal_corporativo.py`

**Patrón**: Router + Multi-Agent Handoff

**Descripción**: Sistema multi-agente para atender consultas legales de empleados de una empresa.

**Características**:
- ✅ Clasificación automática por área legal
- ✅ 5 agentes especializados (laboral, corporativo, contractual, compliance, tributario)
- ✅ Escalamiento a humano cuando es necesario
- ✅ Persistencia de conversaciones
- ✅ Metadata para analytics

**Casos de Uso**:
- Consultas laborales (despidos, prestaciones)
- Consultas corporativas (constitución de empresas)
- Consultas contractuales
- Compliance y normativa
- Consultas tributarias

**Ejecutar**:
```bash
python src/course_examples/casos_reales/01_asistente_legal_corporativo.py
```

**Configuración**:
```python
# Requiere en .env:
GOOGLE_API_KEY=tu_api_key_aqui
```

---

### 2. Clasificador y Enrutador de Documentos

**Archivo**: `02_clasificador_documentos.py`

**Patrón**: Parallelization + Routing

**Descripción**: Sistema para clasificar documentos legales y enrutarlos al proceso especializado adecuado.

**Características**:
- ✅ Clasificación de 9 tipos de documentos
- ✅ Extracción de elementos clave
- ✅ Asignación de proceso especializado
- ✅ Generación de informes automáticos
- ✅ Priorización por urgencia

**Tipos de Documentos**:
- Tutelas
- Demandas
- Sentencias
- Contratos
- Leyes
- Decretos
- Derechos de petición
- Actos administrativos
- Otros

**Procesos Especializados**:
- Proceso de Tutela (30 min)
- Proceso Contencioso (60 min)
- Análisis de Fallo (45 min)
- Revisión Contractual (40 min)
- Investigación Normativa (20 min)
- Respuesta a Petición (15 min)

**Ejecutar**:
```bash
python src/course_examples/casos_reales/02_clasificador_documentos.py
```

---

## 🎯 Cómo Usar en Producción

### Paso 1: Adaptar a tu Caso

```python
# Ejemplo: Adaptar asistente legal para firma específica

# 1. Modificar agentes especializados
AGENTES = {
    "mi_especialidad_1": AgenteEspecializado(
        area="mi_especialidad_1",
        system_prompt="""Tu prompt especializado aquí"""
    ),
    # ... más agentes
}

# 2. Modificar clasificación
prompt_clasificacion = ChatPromptTemplate.from_messages([
    ("system", """Tu prompt de clasificación aquí"""),
    ("user", "Consulta: {consulta}")
])
```

### Paso 2: Configurar Persistencia

```python
# En producción, usar base de datos real
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(
    conn_string="postgresql://user:pass@localhost:5432/dbname"
)

graph = builder.compile(checkpointer=checkpointer)
```

### Paso 3: Integrar con Sistemas Existentes

```python
# Ejemplo: Integrar con sistema de tickets
def crear_ticket(consulta, respuesta):
    """Crear ticket en sistema de gestión."""
    
    # Tu integración aquí
    response = requests.post(
        "https://tu-sistema.com/api/tickets",
        json={
            "consulta": consulta,
            "respuesta": respuesta.respuesta,
            "area": respuesta.area
        }
    )
    
    return response.json()
```

### Paso 4: Monitoreo y Logging

```python
# Configurar logging para producción
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agente_legal.log', encoding='utf-8'),
        logging.handlers.RotatingFileHandler(
            'agente_legal.log',
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
    ]
)
```

---

## 📊 Métricas y KPIs

### Para Asistente Legal Corporativo

| Métrica | Fórmula | Meta |
|---------|---------|------|
| **Precisión de Clasificación** | (Clasificaciones correctas / Total) × 100 | >90% |
| **Tasa de Escalamiento** | (Consultas escaladas / Total) × 100 | <20% |
| **Tiempo Promedio Respuesta** | Suma tiempos / Total consultas | <30 segundos |
| **Satisfacción Usuario** | Encuestas post-consulta | >4.0/5.0 |
| **Costo por Consulta** | Costo total / Total consultas | <$0.20 |

### Para Clasificador de Documentos

| Métrica | Fórmula | Meta |
|---------|---------|------|
| **Precisión de Clasificación** | (Documentos bien clasificados / Total) × 100 | >95% |
| **Tiempo Promedio Procesamiento** | Suma tiempos / Total documentos | <2 minutos |
| **Elementos Extraídos** | (Elementos extraídos / Esperados) × 100 | >80% |
| **Tasa de Procesamiento Exitoso** | (Procesos exitosos / Total) × 100 | >98% |

---

## 🔧 Personalización Avanzada

### Agregar Nuevo Agente Especializado

```python
# 1. Definir el agente
AGENTES["nueva_area"] = AgenteEspecializado(
    area="nueva_area",
    system_prompt="""Eres un experto en nueva área.
    
    Tus capacidades:
    - Capacidad 1
    - Capacidad 2
    
    Responde citando normativa relevante.
    """
)

# 2. Actualizar clasificación
prompt_clasificacion = ChatPromptTemplate.from_messages([
    ("system", """...
    
    Áreas disponibles:
    - laboral: ...
    - corporativo: ...
    - nueva_area: Descripción de la nueva área
    ...
    """),
    ("user", "Consulta: {consulta}")
])
```

### Agregar Nuevo Tipo de Documento

```python
# 1. Actualizar modelo
class ClasificacionDocumento(BaseModel):
    tipo: Literal[
        "tutela", 
        "demanda",
        # ... tipos existentes
        "nuevo_tipo"  # ← Agregar aquí
    ]

# 2. Agregar proceso especializado
class ProcesosEspecializados:
    @staticmethod
    def procesar_nuevo_tipo(documento, elementos):
        """Procesar nuevo tipo de documento."""
        
        informe = "📄 INFORME DE NUEVO TIPO\n"
        # ... lógica de procesamiento
        
        return informe

# 3. Actualizar routing
procesos = {
    # ... procesos existentes
    "nuevo_tipo": {
        "proceso": "Nuevo Proceso",
        "prioridad": "media",
        "tiempo": "30",
        "responsable": "Abogado Especialista"
    }
}
```

---

## 🚀 Deploy a Producción

### Opción 1: API REST con FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ConsultaRequest(BaseModel):
    pregunta: str
    empleado_id: str
    departamento: str = "general"
    urgencia: str = "media"

@app.post("/api/consultar")
async def consultar(request: ConsultaRequest):
    sistema = AsistenteLegalCorporativo()
    respuesta = sistema.consultar(
        pregunta=request.pregunta,
        empleado_id=request.empleado_id,
        departamento=request.departamento,
        urgencia=request.urgencia
    )
    
    return {
        "respuesta": respuesta.respuesta,
        "area": respuesta.area,
        "confianza": respuesta.nivel_confianza
    }

# Ejecutar
# uvicorn api:app --host 0.0.0.0 --port 8000
```

### Opción 2: Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Opción 3: Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: asistente-legal
spec:
  replicas: 3
  selector:
    matchLabels:
      app: asistente-legal
  template:
    metadata:
      labels:
        app: asistente-legal
    spec:
      containers:
      - name: asistente-legal
        image: tu-registry/asistente-legal:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: google-api-key
```

---

## 📚 Recursos Adicionales

### Documentación
- [LangGraph Multi-Agent](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [LangGraph Routing](https://docs.langchain.com/oss/python/langgraph/workflows-agents#routing)
- [Production Patterns](https://towardsai.net/p/machine-learning/production-ready-ai-agents-8-patterns-that-actually-work)

### Casos de Éxito
- **Bank of America**: 8 patrones de agentes en producción
- **Coinbase**: Multi-agent para soporte al cliente
- **UiPath**: Orchestrator-workers para automatización

---

## 🤝 Contribuciones

Los casos reales están abiertos a contribuciones. Para agregar un nuevo caso:

1. Crear archivo `03_nuevo_caso.py`
2. Seguir estructura de casos existentes
3. Incluir documentación completa
4. Agregar ejemplos de uso
5. Submitir pull request

---

*Creado: 2026-03-29*  
*Última actualización: 2026-03-29*  
*Mantenimiento: Equipo del Curso*
