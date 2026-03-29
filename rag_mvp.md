# 📋 INFORME TÉCNICO - RAG MVP (LegalRAG)

## 1. RESUMEN EJECUTIVO

**Proyecto**: RAG MVP (Minimum Viable Product)  
**Ubicación**: `C:\Users\DELL\Desktop\Software Fenix\RAG MVP`  
**Propósito**: Sistema de Retrieval-Augmented Generation (RAG) especializado en documentos legales colombianos  
**Estado**: En desarrollo inicial (estructura base implementada)

---

## 2. ARQUITECTURA DEL PROYECTO

### 2.1 Estructura de Directorios

```
RAG MVP/
├── main.py                    # Punto de entrada principal
├── pyproject.toml             # Configuración del proyecto (UV)
├── requirements.txt           # Dependencias Python
├── docker-compose.yml         # Orquestación de contenedores (Qdrant)
├── .python-version            # Versión Python: 3.12
├── .gitignore                 # Reglas de exclusión Git
├── README.md                  # Documentación (vacío actualmente)
│
├── src/                       # Código fuente principal
│   ├── __init__.py
│   ├── agent/                 # Lógica de agentes (vacío)
│   ├── indexing/              # Indexación (vacío)
│   ├── ingestion/             # Ingesta de documentos
│   │   ├── __init__.py
│   │   ├── ingestion.py       # Procesamiento con LlamaParse
│   │   └── loader.py          # Cargador de documentos (incompleto)
│   ├── retrieval/             # Recuperación (vacío)
│   ├── tools/                 # Herramientas (vacío)
│   └── utils/                 # Utilidades
│       ├── loggers.py         # Sistema de logging
│       └── preproccesor.py    # Pre-procesamiento de imágenes
│
├── data/                      # Datos
│   ├── sample.pdf             # PDF de ejemplo
│   ├── input/                 # Documentos de entrada (vacío)
│   └── processed/             # Documentos procesados (vacío)
│
├── storage/                   # Almacenamiento de vectores
├── logs/                      # Logs de ejecución
├── test/                      # Tests unitarios
└── doc/                       # Documentación de referencia (10 subdirectorios)
```

---

## 3. STACK TECNOLÓGICO

### 3.1 Dependencias Principales

| Categoría | Librerías | Versión |
|-----------|-----------|---------|
| **Framework RAG** | llama-index | >=0.14.19 |
| **LLM** | google-generativeai, langchain-google-genai | >=0.8.6, >=4.2.1 |
| **Orquestación** | langgraph, langchain | >=1.1.3, >=1.2.13 |
| **Vector Stores** | ChromaDB, Qdrant | - |
| **Parsing** | llama-parse | >=0.6.94 |
| **PDF** | PyMuPDF, PyPDF2, pdf2image | >=1.27.2.2, >=3.0.0, >=1.17.0 |
| **OCR** | pytesseract, opencv-python | >=0.3.13, >=4.13.0.92 |
| **Reranking** | flashrank | >=0.2.10 |
| **MCP** | langchain-mcp-adapters | >=0.2.2 |

### 3.2 Infraestructura

- **Base de Datos Vectorial**: Qdrant (vía Docker)
  - Puerto HTTP: 6333
  - Puerto gRPC: 6334
  - Volumen persistente: `./qdrant_storage`

- **Python**: 3.12 (gestionado por UV)
- **Package Manager**: UV (moderno, rápido)

---

## 4. COMPONENTES IMPLEMENTADOS

### 4.1 Módulo de Ingestión (`src/ingestion/`)

#### `ingestion.py`
**Funcionalidad**: Procesamiento de documentos con LlamaParse

```python
def process_documents(directory_path):
    # Lee documentos del directorio
    # Procesa cada documento con LlamaParse
    # Retorna documentos procesados en formato Markdown
```

**Características**:
- Parser de tablas en español
- Formato de salida: Markdown
- Integración con LlamaParse API
- Requiere variable de entorno: `LLAMA_PARSE_API_KEY`

#### `loader.py` (Incompleto)
**Estado**: Solo imports inicializados

**Dependencias declaradas**:
- `PyPDF2`, `fitz (PyMuPDF)` - Lectura de PDF
- `pytesseract` - OCR
- `tqdm` - Barras de progreso
- `ImagePreprocessor` - Pre-procesamiento de imágenes

### 4.2 Módulo de Utilidades (`src/utils/`)

#### `loggers.py`
**Funcionalidad**: Sistema de logging dual (archivo + consola)

**Características**:
- Logs con fecha en el nombre: `app_YYYYMMDD.log`
- Directorio configurable vía `LOGS_PATH`
- Formato: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Codificación UTF-8
- Handlers: FileHandler + StreamHandler

#### `preproccesor.py` ⭐ (Componente más desarrollado)
**Clase**: `ImagePreprocessor`

**Métodos**:

| Método | Propósito |
|--------|-----------|
| `deskew(image)` | Enderezar texto en PDFs escaneados |
| `enhance_for_ocr(image_path, save_debug)` | Pipeline completo de mejora para OCR |
| `pdf_to_images(pdf_path, dpi=400)` | Conversión PDF a imágenes de alta calidad |

**Pipeline de Pre-procesamiento OCR**:
1. Escala de grises
2. Deskew (enderezar con `cv2.minAreaRect`)
3. Binarización Otsu + inversa
4. Eliminación de ruido (`fastNlMeansDenoising`)
5. Mejora de contraste local (CLAHE)
6. Operaciones morfológicas (MORPH_CLOSE)

**Configuración específica para documentos legales colombianos**:
- DPI: 400 (alta calidad para OCR legal)
- Debug: Guardado de imágenes procesadas en `debug_images/`

---

## 5. COMPONENTES PENDIENTES DE IMPLEMENTAR

Los siguientes directorios están creados pero **vacíos**:

| Directorio | Propósito Esperado |
|------------|-------------------|
| `src/agent/` | Agentes LangGraph para razonamiento |
| `src/indexing/` | Indexación en vector store |
| `src/retrieval/` | Estrategias de recuperación |
| `src/tools/` | Herramientas para agentes |

---

## 6. DOCUMENTACIÓN DE REFERENCIA (`doc/`)

La carpeta `doc/` contiene una **biblioteca exhaustiva** de implementaciones de referencia:

### 6.1 Volumen de Documentación

| Tipo | Cantidad |
|------|----------|
| README.md | 15+ |
| Jupyter Notebooks | 100+ |
| Scripts Python | 200+ |
| Imágenes/Diagramas | 120+ |

### 6.2 Temas Cubiertos

#### Técnicas RAG (35+)
- Simple RAG, Agentic RAG, Self-RAG, CRAG
- Fusion Retrieval, Adaptive Retrieval
- Graph RAG, Microsoft GraphRAG
- RAPTOR, Semantic Chunking
- Contextual Compression, Reranking
- HyDe (Hypothetical Document Embedding)
- Multi-model RAG

#### Patrones Agénticos
- Reflection Pattern
- Tool Call Pattern
- ReAct Planning Pattern
- Multi-Agent Pattern
- Supervisor Architecture
- Swarm Intelligence

#### LangGraph Features
- State Management
- Conditional Routing
- Human-in-the-Loop
- Memory & Persistence
- Streaming
- Subgraphs
- Tool Calling

#### MCP (Model Context Protocol)
- Server-Client Architecture
- Multiple Server Support
- SSE Transport
- HTTP Streaming

### 6.3 Proyectos Completos de Referencia

1. **Advanced-RAG-LangGraph-main**: App web con Streamlit + ChromaDB
2. **agentic_patterns**: Implementación de patrones multi-agente
3. **all_rag_techniques**: 35 notebooks con técnicas RAG
4. **examples**: 65+ ejemplos de LangGraph
5. **LangGraphProjects-main**: 20 capítulos del libro "The Complete LangGraph Blueprint"
6. **LangGraphRAG-main**: Sistema RAG terminal-based

---

## 7. ESTADO ACTUAL DEL PROYECTO

### 7.1 Completado ✅
- [x] Estructura de directorios
- [x] Configuración de dependencias (pyproject.toml, requirements.txt)
- [x] Docker Compose para Qdrant
- [x] Sistema de logging
- [x] Pre-procesador de imágenes para OCR
- [x] Parser de documentos con LlamaParse

### 7.2 Pendiente 🔲
- [ ] Completar `loader.py`
- [ ] Implementar módulo de indexación
- [ ] Implementar módulo de retrieval
- [ ] Implementar agentes LangGraph
- [ ] Implementar herramientas
- [ ] Tests unitarios
- [ ] Documentación del proyecto (README.md vacío)
- [ ] Variables de entorno (.env no existe)

### 7.3 Variables de Entorno Requeridas
```bash
LLAMA_PARSE_API_KEY=<tu_api_key>
LOGS_PATH=<ruta_para_logs>
# Probablemente también se necesitará:
GOOGLE_API_KEY=<tu_api_key_google>
QDRANT_URL=http://localhost:6333
```

---

## 8. RECOMENDACIONES TÉCNICAS

### 8.1 Prioridades de Desarrollo

1. **Completar `loader.py`**: Implementar la lógica de carga que usa los imports ya declarados
2. **Configurar variables de entorno**: Crear `.env` con las API keys necesarias
3. **Implementar indexación**: Conectar con Qdrant y crear los índices vectoriales
4. **Implementar retrieval**: Estrategias de búsqueda semántica + reranking
5. **Crear agente LangGraph**: Orquestar el flujo RAG

### 8.2 Aprovechar la Documentación

La carpeta `doc/` contiene implementaciones completas que pueden servir como referencia:
- `doc/Advanced-RAG-LangGraph-main/` → Arquitectura similar
- `doc/all_rag_techniques/` → Técnicas para evaluar
- `doc/LangGraphRAG-main/` → Implementación terminal-based

### 8.3 Consideraciones para Documentos Legales

El pre-procesador está optimizado para documentos legales colombianos:
- Alto DPI (400) para preservar detalles
- Deskew para documentos escaneados
- CLAHE para mejorar contraste en textos antiguos
- Debug mode para validar calidad

---

## 9. DIAGRAMA DE ARQUITECTURA PROPUESTA

```
┌─────────────────────────────────────────────────────────────┐
│                      USUARIO QUERY                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGENTE LANGGRAPH                         │
│  (Orquestador: routing, memory, tool calling)               │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   MÓDULO INDEXACIÓN   │       │   MÓDULO RETRIEVAL    │
│  - PDF → Imágenes     │       │  - Búsqueda vectorial │
│  - OCR (Tesseract)    │       │  - Reranking          │
│  - LlamaParse         │       │  - Fusion retrieval   │
│  - Embeddings (Google)│       │  - Query expansion    │
│  - Qdrant Store       │       │                       │
└───────────────────────┘       └───────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    QDRANT VECTOR STORE                      │
│              (Embeddings de documentos legales)             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    GENERACIÓN RESPUESTA                     │
│  - Google Gemini                                            │
│  - Contexto recuperado + query                              │
│  - Respuesta en español                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. CONCLUSIONES

1. **El proyecto está en fase inicial**: La estructura base está bien organizada pero los componentes críticos faltan implementación.

2. **Excelente documentación de referencia**: La carpeta `doc/` contiene suficiente material para implementar todas las funcionalidades necesarias.

3. **Especialización clara**: El pre-procesador de imágenes revela que el sistema está diseñado para documentos legales colombianos escaneados.

4. **Stack tecnológico moderno**: UV, LangGraph, Qdrant, Google Gemini son tecnologías actuales y apropiadas para el caso de uso.

5. **Próximos pasos críticos**:
   - Completar el loader de documentos
   - Implementar indexación y retrieval
   - Configurar credenciales
   - Desarrollar el agente principal

---

*Informe generado: 2026-03-29*
