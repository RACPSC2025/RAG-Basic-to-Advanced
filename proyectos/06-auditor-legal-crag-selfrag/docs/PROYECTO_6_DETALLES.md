# ⚖️ Proyecto 06: Auditor Legal Inteligente (CRAG + Self-RAG)

## 1. RESUMEN DEL PROYECTO
**Nombre Real**: Auditor Legal Inteligente con Recuperación Correctiva y Auto-Reflexión.
**Propósito**: Crear un sistema RAG de grado profesional para el dominio legal colombiano que garantice 0% de alucinaciones mediante la validación continua de hechos y relevancia.

---

## 2. ARQUITECTURA TÉCNICA (Stack Actualizado)
- **Orquestación**: [LangGraph](https://langchain-ai.github.io/langgraph/) (Grafos de estado cíclicos).
- **LLM**: Google Gemini 2.5 Flash (Configurado vía .env).
- **Vector Store**: ChromaDB (Local por defecto) / Qdrant Cloud (Ready).
- **Ingestión**: LlamaParse (Especializado en tablas legales colombianas).
- **Visualización**: Rich (Interfaz CLI profesional con métricas en tiempo real).

---

## 3. FASES DE DESARROLLO Y DOCUMENTACIÓN (Actualizado)

### Fase 1: Ingestión con Observabilidad
- **Descripción**: Procesamiento de documentos con monitoreo de rendimiento.
- **Métricas**: 
  - `Parsing Speed`: Documentos procesados por minuto.
  - `Embedding Throughput`: Chunks/segundo (Latencia de API).
  - `IO Latency`: Tiempo de persistencia en disco.

### Fase 2: Auditoría de Tokens (Token Management)
- **Lógica**: Seguimiento local del consumo para evitar bloqueos en el Free Tier de Gemini.
- **Estimación**: 1 token por cada 3.5 caracteres (ajustado para español legal).

### Fase 2: Calificación de Documentos (Document Grading)
- **Lógica**: Un nodo de LangGraph evalúa cada documento recuperado: `Relevante` o `Irrelevante`.
- **Salida**: Si no hay documentos relevantes, el flujo se desvía hacia la **Transformación de Consulta**.

### Fase 3: Recuperación Correctiva (CRAG)
- **Lógica**: Si los documentos iniciales no sirven, el LLM optimiza la búsqueda legal (Query Transformation) para intentar una nueva recuperación más precisa.
- **Diferenciador**: Evita que el LLM intente responder con información insuficiente.

### Fase 4: Generación y Verificación de Alucinaciones (Self-RAG)
- **Lógica**: 
  1. **Generación**: Basada únicamente en documentos calificados como relevantes.
  2. **Hallucination Grader**: ¿La respuesta está soportada por los documentos? (Grounding).
  3. **Answer Grader**: ¿La respuesta resuelve realmente la duda del usuario?

### Fase 5: Interfaz de Auditoría y Logs
- **Descripción**: Sistema de logging que muestra al usuario el "razonamiento" del auditor (ej. "Descarté el Doc A por irrelevancia, reescribí la consulta a X...").

---

## 4. ESTRUCTURA DE CÓDIGO (Roadmap)

```bash
proyectos/06-auditor-legal-crag-selfrag/
├── src/
│   ├── agent/
│   │   ├── state.py      # Definición del estado del grafo (TypedDict)
│   │   ├── nodes.py      # Lógica de cada nodo (Retrieve, Grade, Generate)
│   │   └── graph.py      # Construcción del grafo y aristas condicionales
│   ├── config.py         # Variables de entorno y configuración
│   └── utils.py          # Formateo de documentos y prompts
├── docs/
│   └── PROYECTO_6_DETALLES.md # Este documento
├── main.py               # Punto de entrada
└── README.md             # Instrucciones de uso rápido
```

---

## 5. CÓDIGO DE REFERENCIA (Patrón de Diseño Profesional)

### Fragmento de Calificador de Documentos (Pydantic)
```python
class GradeDocuments(BaseModel):
    """Calificación binaria de relevancia de documentos."""
    binary_score: str = Field(description="Documentos relevantes 'yes' o 'no'")
```

### Lógica de Re-intentos
El grafo permite hasta 3 reintentos de búsqueda antes de admitir que no tiene la información, garantizando honestidad técnica.

---

*Proyecto diseñado por Gemini CLI - 2026-03-29*
