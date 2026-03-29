# 📚 Curso Completo: LangChain + LangGraph para RAG

## Descripción
Curso práctico paso a paso para dominar LangChain y LangGraph, construyendo un sistema RAG (Retrieval-Augmented Generation) desde los fundamentos hasta características avanzadas.

## Stack Tecnológico
- **LLM**: Google Gemini (API gratuita)
- **Framework**: LangChain + LangGraph
- **Vector Store**: Qdrant
- **Lenguaje**: Python 3.12+

---

# 📖 TEMARIO DEL CURSO

## **MÓDULO 0: Configuración del Entorno**
- 0.1 Instalación de dependencias
- 0.2 Configuración de variables de entorno
- 0.3 Estructura del proyecto
- 0.4 Primeros pasos con LangChain

---

## **MÓDULO 1: Fundamentos de LangChain**
### 1.1 Conexión con el LLM (Google Gemini)
- Configuración del modelo
- Primeras llamadas a la API
- Manejo de errores y rate limits

### 1.2 Prompts: Entrada y Estructura
- PromptTemplate básico
- Variables en prompts
- Few-shot prompting
- System prompts vs User prompts

### 1.3 Mensajes y Chat Models
- SystemMessage, HumanMessage, AIMessage
- ChatPromptTemplate
- Manejo de conversaciones

### 1.4 Estructura de Salida
- Output parsers básicos
- PydanticOutputParser
- Structured output con Gemini
- Validación de respuestas

---

## **MÓDULO 2: Memoria y Contexto**
### 2.1 Memoria a Corto Plazo
- ConversationBufferMemory
- ConversationBufferWindowMemory
- Manejo del historial de chat

### 2.2 Memoria a Largo Plazo
- ConversationSummaryMemory
- VectorStoreRetrieverMemory
- Persistencia de conversaciones

### 2.3 Gestión de Contexto
- Token management
- Context window optimization
- Truncamiento inteligente

---

## **MÓDULO 3: Streaming y Experiencia de Usuario**
### 3.1 Streaming de Tokens
- Streaming básico con LangChain
- Manejo de eventos en tiempo real
- UI updates con streaming

### 3.2 Callbacks y Logging
- Custom callbacks
- Debugging con LangSmith
- Performance monitoring

---

## **MÓDULO 4: Introducción a LangGraph**
### 4.1 Conceptos Fundamentales
- ¿Qué es LangGraph?
- Nodes, Edges, State
- Graph vs Sequential chains

### 4.2 Tu Primer Grafo
- StateGraph básico
- Definición de estado
- Primer node y edge

### 4.3 Conditional Routing
- Conditional edges
- Toma de decisiones en el grafo
- Ramificación dinámica

---

## **MÓDULO 5: Herramientas y Function Calling**
### 5.1 Creación de Herramientas
- @tool decorator
- Tool schemas
- Manejo de errores en tools

### 5.2 Tool Calling con LLMs
- Bind tools al modelo
- Tool execution automático
- Multi-tool scenarios

### 5.3 Herramientas Personalizadas
- Búsqueda web
- Cálculos matemáticos
- APIs externas

---

## **MÓDULO 6: Human in the Loop**
### 6.1 Interrupciones y Aprobación
- Breakpoints
- Human approval
- Modificación de estado

### 6.2 Time Travel y Edición
- Editar estado del grafo
- Revertir a estados anteriores
- Fork y replay

---

## **MÓDULO 7: RAG - Retrieval-Augmented Generation**
### 7.1 Fundamentos de RAG
- ¿Qué es RAG y por qué usarlo?
- Arquitectura básica
- Flujo completo

### 7.2 Carga de Documentos
- Document loaders (PDF, TXT, Web)
- Custom loaders
- Error handling

### 7.3 Chunking y Segmentación
- CharacterTextSplitter
- RecursiveCharacterTextSplitter
- Semantic chunking
- Chunk size optimization

### 7.4 Embeddings
- Conceptos de embeddings
- Google Gemini embeddings
- Comparación de modelos

### 7.5 Vector Stores
- Introducción a bases vectoriales
- Qdrant configuración
- CRUD de vectores
- Similarity search

### 7.6 Retrieval Strategies
- Basic retrieval
- Top-k search
- Score threshold
- MMR (Maximal Marginal Relevance)

### 7.7 Reranking
- Contextual compression
- FlashRank integration
- Relevance filtering

---

## **MÓDULO 8: RAG Avanzado con LangGraph**
### 8.1 Agentic RAG
- RAG + Tool calling
- Query routing
- Multi-query retrieval

### 8.2 Advanced Retrieval Patterns
- Fusion retrieval (hybrid search)
- Parent document retriever
- Self-query retrieter

### 8.3 RAG con Memoria
- Conversation + RAG
- Contextual retrieval
- Follow-up questions

### 8.4 Evaluation y Testing
- RAG evaluation metrics
- Faithfulness, relevance, context precision
- Automated testing

---

## **MÓDULO 9: Patrones Avanzados de Agentes**
### 9.1 ReAct Pattern
- Reasoning + Acting
- Plan-and-execute
- Iterative refinement

### 9.2 Multi-Agent Systems
- Agent collaboration
- Supervisor pattern
- Handoff entre agentes

### 9.3 Reflection Pattern
- Self-critique
- Generate + Refine
- Quality improvement loops

---

## **MÓDULO 10: Producción y Deploy**
### 10.1 Persistencia
- Checkpointers
- State persistence
- Postgres integration

### 10.2 Optimización
- Caching de respuestas
- Batch processing
- Rate limit handling

### 10.3 Monitoreo
- LangSmith tracing
- Metrics y alertas
- Performance tuning

---

## **MÓDULO 11: Proyecto Final**
### 11.1 RAG Legal Completo
- Arquitectura completa
- Integración de todos los componentes
- Testing end-to-end

### 11.2 Deployment
- Dockerización
- API con FastAPI
- Frontend opcional (Streamlit)

---

# 📁 Estructura de Archivos del Curso

```
RAG MVP/
├── docs/
│   └── curso/
│       ├── 00-configuracion/
│       │   ├── README.md
│       │   └── codigo/
│       ├── 01-fundamentos-langchain/
│       │   ├── 01-conexion-llm.md
│       │   ├── 02-prompts.md
│       │   ├── 03-mensajes.md
│       │   └── 04-estructura-salida.md
│       ├── 02-memoria/
│       │   ├── 01-corto-plazo.md
│       │   └── 02-largo-plazo.md
│       ├── 03-streaming/
│       ├── 04-langgraph-basico/
│       ├── 05-herramientas/
│       ├── 06-human-in-loop/
│       ├── 07-rag-fundamentos/
│       ├── 08-rag-avanzado/
│       ├── 09-patrones-avanzados/
│       └── 10-produccion/
│
├── src/
│   ├── course_examples/
│   │   ├── modulo_01/
│   │   ├── modulo_02/
│   │   └── ...
│   └── ...
└── ...
```

---

# 🎯 Metodología

Cada lección incluye:
1. **Teoría**: Explicación conceptual clara
2. **Código**: Ejemplo completo y funcional
3. **Ejercicio**: Práctica para reforzar
4. **Recursos**: Links a documentación oficial

---

# 📝 Estado del Curso

| Módulo | Estado | Progreso |
|--------|--------|----------|
| 0. Configuración | Pendiente | 0% |
| 1. Fundamentos LangChain | Pendiente | 0% |
| 2. Memoria | Pendiente | 0% |
| 3. Streaming | Pendiente | 0% |
| 4. LangGraph Básico | Pendiente | 0% |
| 5. Herramientas | Pendiente | 0% |
| 6. Human in the Loop | Pendiente | 0% |
| 7. RAG Fundamentos | Pendiente | 0% |
| 8. RAG Avanzado | Pendiente | 0% |
| 9. Patrones Avanzados | Pendiente | 0% |
| 10. Producción | Pendiente | 0% |
| 11. Proyecto Final | Pendiente | 0% |

---

# 🔗 Recursos Oficiales

- [LangChain Docs](https://docs.langchain.com/oss/python/langchain)
- [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Google Gemini API](https://ai.google.dev/docs)

---

*Curso creado: 2026-03-29*
*Última actualización: 2026-03-29*
