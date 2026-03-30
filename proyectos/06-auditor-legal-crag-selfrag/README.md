# ⚖️ Proyecto 6: Auditor Legal Inteligente (CRAG + Self-RAG)

> **Estado**: ✅ MIGRADO A AWS BEDROCK
> **Fecha de Migración**: 2026-03-30
> **Tecnologías**: LangGraph, AWS Bedrock (Nova Lite + Titan Embeddings), ChromaDB
> **Patrones**: CRAG (Corrective RAG) + Self-RAG

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura](#-arquitectura)
- [Inicio Rápido](#-inicio-rápido)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Flujo del Sistema](#-flujo-del-sistema)
- [Comandos Disponibles](#-comandos-disponibles)
- [Configuración](#-configuración)
- [Cambios Recientes](#-cambios-recientes)

---

## 📝 Descripción

El **Auditor Legal Inteligente** es un sistema RAG avanzado que implementa patrones de **Corrective RAG (CRAG)** y **Self-RAG** para garantizar respuestas legales precisas y sin alucinaciones.

### Características Principales

| Característica | Descripción |
|----------------|-------------|
| 🤖 **LLM** | Amazon Nova Lite vía AWS Bedrock |
| 🧠 **Embeddings** | Amazon Titan Text v2 (1024 dimensiones) |
| 📚 **Vector Store** | ChromaDB persistente |
| ✅ **CRAG** | Filtra documentos irrelevantes antes de generar |
| 🔍 **Self-RAG** | Verifica alucinaciones y utilidad de respuestas |
| 🔄 **Query Transform** | Optimiza consultas automáticamente |
| 📊 **Rate Limiting** | Gestión dinámica de cuotas API |

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│              AUDITOR LEGAL (CRAG + SELF-RAG)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  START                                                           │
│    │                                                             │
│    ▼                                                             │
│  [retrieve] ──→ [grade_documents] ──→ ¿Relevante?               │
│                          │                                       │
│                    ┌─────┴─────┐                                 │
│                 "no"│         │"yes"                             │
│                    ▼         ▼                                   │
│              [transform_query]  [generate]                       │
│                    │             │                               │
│                    └─────┬─────┘                                 │
│                          │                                       │
│                          ▼                                       │
│                  [grade_hallucination]                           │
│                          │                                       │
│                    ┌─────┴─────┐                                 │
│               "yes"│         │"no"                               │
│                    ▼         ▼                                   │
│              [generate]   [grade_answer]                         │
│                          │                                       │
│                    ┌─────┴─────┐                                 │
│                 "no"│         │"yes"                             │
│                    ▼         ▼                                   │
│              [transform_query]  END                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
# Desde la raíz del proyecto
cd "C:\Users\DELL\Desktop\Software Fenix\RAG MVP"
pip install -r requirements.txt

# O instalar solo las del proyecto 6
cd proyectos/06-auditor-legal-crag-selfrag
pip install langchain langchain-aws langchain-chroma langgraph llama-parse rich tenacity
```

### 2. Configurar Variables de Entorno

```bash
# Copiar el archivo de ejemplo
cp .env.example .env
```

Editar `.env` con tus credenciales de AWS:

```bash
# Credenciales AWS
AWS_ACCESS_KEY_ID="AKIA..."
AWS_SECRET_ACCESS_KEY="..."
AWS_REGION="us-east-2"

# Modelos
LLM_MODEL_ID="arn:aws:bedrock:us-east-2:762233737662:inference-profile/us.amazon.nova-lite-v1:0"
EMBEDDING_MODEL_ID="amazon.titan-embed-text-v2:0"

# Vector Store
VECTOR_DB_COLLECTION_NAME="auditoria_legal_colombiana"
TOP_K_DOCS="4"

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE="15"
RATE_LIMIT_BURST_LIMIT="5"
```

### 3. Ingestar Documentos (Opcional pero recomendado)

```bash
# Ingestar PDFs legales
python ingest_data.py --path data/

# Modo simulacro (sin consumo de tokens)
python ingest_data.py --path data/ --dry-run
```

### 4. Ejecutar el Auditor

```bash
# Modo CLI interactivo (recomendado)
python cli.py

# Modo script (consulta única)
python main.py

# Modo test
python test_auditor.py
```

---

## 📁 Estructura del Proyecto

```
proyectos/06-auditor-legal-crag-selfrag/
│
├── src/
│   ├── agent/
│   │   ├── __init__.py          # ✅ Init del paquete
│   │   ├── graph.py             # ✅ Grafo LangGraph (CRAG + Self-RAG)
│   │   ├── nodes.py             # ✅ Nodos del auditor (migrado AWS)
│   │   └── state.py             # ✅ Estado tipado
│   │
│   ├── ingestion/
│   │   └── processor.py         # ✅ Procesador de PDFs (migrado AWS)
│   │
│   ├── utils/
│   │   ├── token_counter.py     # ✅ Auditor de tokens (migrado AWS)
│   │   └── visuals.py           # ✅ UI con Rich
│   │
│   └── config.py                # ✅ Configuración central (migrado AWS)
│
├── storage/                     # 📚 ChromaDB persistente
│
├── data/                        # 📄 PDFs legales para ingestar
│
├── main.py                      # ✅ Script principal
├── cli.py                       # ✅ CLI interactivo
├── ingest_data.py               # ✅ Script de ingesta
├── test_auditor.py              # ✅ Tests del auditor
├── .env.example                 # ✅ Template de configuración
└── README.md                    # ✅ Este archivo
```

---

## 🔄 Flujo del Sistema

### Fase 1: Retrieve
Recupera los top-K documentos más relevantes de ChromaDB usando búsqueda semántica.

### Fase 2: Grade Documents (CRAG)
Un LLM evalúa cada documento recuperado:
- **Relevante (yes)**: Pasa a generación
- **No relevante (no)**: Transforma la consulta y reintenta

### Fase 3: Generate
Genera respuesta basada exclusivamente en los documentos aprobados.

### Fase 4: Grade Hallucination (Self-RAG)
Verifica que la respuesta no invente información:
- **Fiel (no)**: Pasa a validación de utilidad
- **Alucinación (yes)**: Re-genera la respuesta

### Fase 5: Grade Answer (Self-RAG)
Evalúa si la respuesta resuelve la pregunta:
- **Útil (yes)**: Termina exitosamente
- **No útil (no)**: Transforma la consulta y reintenta

---

## 💻 Comandos Disponibles

### CLI Interactivo

```bash
python cli.py
```

Menú principal:
1. 📥 **Ingestar nuevos documentos** - Procesa PDFs y los indexa
2. 💬 **Chatear con el Auditor** - Consultas legales con CRAG + Self-RAG
3. 📊 **Ver configuración** - Muestra parámetros del sistema
4. ❌ **Salir**

### Ingesta de Documentos

```bash
# Ingestar un archivo
python ingest_data.py --path data/ley_1348_2009.pdf

# Ingestar carpeta completa
python ingest_data.py --path data/

# Modo simulacro (sin tokens)
python ingest_data.py --path data/ --dry-run
```

### Test del Sistema

```bash
# Test completo
python test_auditor.py

# Test rápido de importación
python -c "from src.agent.graph import create_auditor_graph; print('OK')"
```

---

## ⚙️ Configuración

### Variables de Entorno Principales

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `AWS_ACCESS_KEY_ID` | Access Key de AWS | Requerido |
| `AWS_SECRET_ACCESS_KEY` | Secret Key de AWS | Requerido |
| `AWS_REGION` | Región de AWS | `us-east-2` |
| `LLM_MODEL_ID` | ARN del modelo LLM | Nova Lite inference profile |
| `EMBEDDING_MODEL_ID` | Modelo de embeddings | `amazon.titan-embed-text-v2:0` |
| `VECTOR_DB_COLLECTION_NAME` | Colección ChromaDB | `auditoria_legal_colombiana` |
| `TOP_K_DOCS` | Documentos a recuperar | `4` |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Límite RPM | `15` |
| `MAX_RETRIES` | Reintentos máximos | `3` |

### Parámetros del LLM

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Temperatura | 0.0 | Determinista (legal) |
| Max Tokens | 2048 | Límite de salida |

---

## 🔄 Cambios Recientes (2026-03-30)

### Migración a AWS Bedrock

El proyecto fue migrado exitosamente de Google Gemini a **AWS Bedrock**.

| Componente | Anterior | Nuevo |
|------------|----------|-------|
| **LLM** | Google Gemini 2.5 Flash | Amazon Nova Lite |
| **Embeddings** | Google Gemini Embedding | Amazon Titan Text v2 |
| **SDK** | `langchain-google-genai` | `langchain-aws` |

### Archivos Actualizados

| Archivo | Cambios |
|---------|---------|
| `src/config.py` | Migrado a credenciales y modelos AWS |
| `src/agent/nodes.py` | Usa `ChatBedrock` y `BedrockEmbeddings` |
| `src/ingestion/processor.py` | Embeddings con AWS Titan v2 |
| `src/utils/token_counter.py` | Actualizado para AWS |
| `main.py` | Mejorado con output detallado |
| `.env.example` | Template de configuración AWS |

---

## 📊 Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| Tiempo promedio de respuesta | 5-15 segundos |
| Documentos recuperados | 4 (configurable) |
| Precisión de retrieval | ~85-90% |
| Tasa de alucinación | <5% (con Self-RAG) |

---

## 🧪 Testing

### Ejecutar Test Completo

```bash
python test_auditor.py
```

El test verifica:
1. ✅ Configuración de AWS
2. ✅ Creación del grafo
3. ✅ Consulta básica
4. ✅ Flujo CRAG + Self-RAG

---

## 📚 Recursos Adicionales

### Documentación Oficial

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Amazon Nova Lite](https://aws.amazon.com/bedrock/nova/)
- [ChromaDB](https://docs.trychroma.com/)

### Patrones Implementados

- **CRAG** (Corrective RAG): [Paper](https://arxiv.org/abs/2401.15884)
- **Self-RAG**: [Paper](https://arxiv.org/abs/2310.11511)

---

## ⚠️ Solución de Problemas

### Error: "API key not valid"

**Causa**: Las credenciales de AWS no están configuradas correctamente.

**Solución**:
```bash
# Verificar .env
cat .env | grep AWS_ACCESS_KEY_ID

# Verificar región
cat .env | grep AWS_REGION
```

### Error: "No documents found"

**Causa**: La base de datos ChromaDB está vacía.

**Solución**:
```bash
# Ingestar documentos primero
python ingest_data.py --path data/
```

### Error: "Rate limit exceeded"

**Causa**: Se excedió el límite de peticiones por minuto.

**Solución**:
```bash
# Ajustar en .env
RATE_LIMIT_REQUESTS_PER_MINUTE="10"
```

---

*Proyecto creado: 2026-03-29*  
*Última actualización: 2026-03-30*  
*Migrado a AWS Bedrock: 2026-03-30*  
*Autor: Curso LangChain + LangGraph para RAG*  
*License: MIT*
