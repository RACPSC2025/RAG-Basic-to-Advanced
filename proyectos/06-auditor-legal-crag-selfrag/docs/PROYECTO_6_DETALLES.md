# ⚖️ Proyecto 06: Auditor Legal Inteligente (CRAG + Self-RAG)

## 1. RESUMEN DEL PROYECTO
**Nombre Real**: Auditor Legal Inteligente con Recuperación Correctiva y Auto-Reflexión.
**Propósito**: Sistema RAG de grado profesional que garantiza 0% de alucinaciones mediante validación continua de hechos y relevancia legal.

---

## 2. ARQUITECTURA TÉCNICA (Stack)
- **Orquestación**: [LangGraph](https://langchain-ai.github.io/langgraph/).
- **LLM**: Google Gemini 1.5 Flash (Optimizado para cuotas Free Tier).
- **Vector Store**: ChromaDB (Persistente local).
- **Observabilidad**: Rich (Métricas de velocidad y consumo en tiempo real).

---

## 3. GESTIÓN DINÁMICA DE INFRAESTRUCTURA (.env)
El sistema está diseñado para ser agnóstico a los límites de la API. Se configura mediante las siguientes variables:

| Variable | Propósito | Efecto en el Código |
| :--- | :--- | :--- |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Control de cuota RPM | Calcula automáticamente el `delay` entre pasos del agente. |
| `RATE_LIMIT_BURST_LIMIT` | Control de ráfagas | Define el tamaño base de los lotes de procesamiento. |
| `INGESTION_BATCH_SIZE` | Tamaño de lote de ingesta | Cuántos documentos se envían a embedding en una sola llamada. |

---

## 4. FASES DEL PIPELINE

### Fase 1: Ingesta Resiliente
Usa **Batching** y **Exponential Backoff** (vía `tenacity`) para subir documentos sin activar el error 429.

### Fase 2: Recuperación Correctiva (CRAG)
El agente evalúa los documentos recuperados. Si son irrelevantes, **optimiza la consulta** y reintenta la búsqueda.

### Fase 3: Auto-Reflexión (Self-RAG)
Incluye dos filtros críticos:
1.  **Hallucination Grader**: Verifica que la respuesta no invente leyes (Grounding).
2.  **Answer Grader**: Asegura que la respuesta sea útil para el caso legal.

---

## 5. MANUAL DE OPERACIÓN

### Ingesta de Documentos:
```bash
python proyectos/06-auditor-legal-crag-selfrag/ingest_data.py --path data/sample.pdf
```

### Ejecución del Auditor:
```bash
python proyectos/06-auditor-legal-crag-selfrag/main.py
```

---
*Documentación actualizada por Aether (Gemini CLI) - 2026-03-29*
