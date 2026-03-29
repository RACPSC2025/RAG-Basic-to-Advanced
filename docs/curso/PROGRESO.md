# 📚 Curso LangChain + LangGraph para RAG - Progreso

## Última Actualización: 2026-03-29

---

## 📊 Estado del Curso

| Módulo | Lecciones | Estado | Archivos |
|--------|-----------|--------|----------|
| **0. Configuración** | 1 | ✅ Completo | 1 docs, 1 código |
| **1. Fundamentos LangChain** | 4 | ✅ Completo | 4 docs, 4 códigos |
| **2. Memoria** | 2 | ✅ Completo | 2 docs, 1 código |
| **3. Streaming** | 1 | ✅ Completo | 1 doc, 1 código |
| **4. LangGraph Básico** | 1 | ✅ Completo | 1 doc, 1 código |
| **5. Herramientas** | 3 | ✅ Completo | 1 docs, 3 códigos |
| **6. Human in the Loop** | 1 | ✅ Completo | 1 doc, 1 código |
| **7. Ejercicios Resueltos** | 1 | ✅ Completo | 1 doc, 1 código |
| **8. Mejores Prácticas** | 1 | ✅ Completo | 1 doc |
| **9. RAG Fundamentos** | 1 | ✅ Completo | 1 doc, 1 código |
| **10. RAG Avanzado** | 1 | ✅ Completo | 1 doc, 1 código |
| **11. Patrones Avanzados** | 1 | ✅ Completo | 1 doc, 1 código |
| **12. Casos Reales Producción** | 1 | ✅ Completo | 1 doc, 2 códigos |
| **13. Producción y Deploy** | 1 | ✅ Completo | 1 doc, 3 códigos |
| **14. Proyecto Final** | 0 | ⏳ Pendiente | - |

**Progreso Total: 93% (13/14 módulos completos)**

---

## 📁 Estructura Creada

```
RAG MVP/
├── docs/curso/
│   ├── README.md                              # ✅ Temario completo
│   │
│   ├── 00-configuracion/
│   │   ├── README.md                          # ✅ Configuración del entorno
│   │   └── codigo/                            # (el código va en src/)
│   │
│   ├── 01-fundamentos-langchain/
│   │   ├── 01-conexion-llm.md                 # ✅ Conexión con Gemini
│   │   ├── 02-prompts.md                      # ✅ Prompt templates
│   │   ├── 03-mensajes.md                     # ✅ Chat models y mensajes
│   │   ├── 04-estructura-salida.md            # ✅ Output parsers
│   │   └── codigo/
│   │
│   ├── 02-memoria/
│   │   ├── 01-corto-plazo.md                  # ✅ Buffer/Window memory
│   │   ├── 02-largo-plazo.md                  # ✅ Summary memory + persistencia
│   │   └── codigo/
│   │
│   ├── 03-streaming/
│   │   ├── 01-streaming-tokens.md             # ✅ Streaming de tokens
│   │   └── codigo/
│   │
│   └── 04-langgraph-basico/
│       ├── 01-primer-grafo.md                 # ✅ StateGraph, nodes, edges
│       └── codigo/
│
├── src/course_examples/
│   ├── modulo_00/
│   │   └── 00_hello_langchain.py              # ✅ Primer script
│   │
│   ├── modulo_01/
│   │   ├── 00_hello_langchain.py              # ✅ Verificación
│   │   ├── 01_conexion_llm.py                 # ✅ Parámetros del modelo
│   │   ├── 02_prompts.py                      # ✅ Prompt templates
│   │   ├── 03_mensajes.py                     # ✅ Mensajes y historial
│   │   └── 04_structured_output.py            # ✅ Output parsers
│   │
│   ├── modulo_02/
│   │   └── 01_memoria_corto_plazo.py          # ✅ Buffer/Window memory
│   │
│   ├── modulo_03/
│   │   └── 01_streaming.py                    # ✅ Streaming de tokens
│   │
│   └── modulo_04/
│       └── 01_primer_grafo.py                 # ✅ LangGraph básico
│
└── rag_mvp.md                                 # ✅ Informe técnico del proyecto
```

---

## 📖 Resumen de Contenido por Módulo

### Módulo 0: Configuración ✅
- Instalación de dependencias con UV/pip
- Configuración de variables de entorno
- Obtención de API Key de Google Gemini
- Primer script de prueba "Hello LangChain"
- Solución de problemas comunes

### Módulo 1: Fundamentos de LangChain ✅

#### 1.1 Conexión con el LLM
- ChatGoogleGenerativeAI initialization
- Parámetros: model, temperature, max_tokens, top_p, timeout
- Manejo de errores y rate limits
- Comparación de modelos Gemini

#### 1.2 Prompts
- PromptTemplate básico
- ChatPromptTemplate con roles
- Few-shot prompting
- System vs User prompts
- Templates reutilizables

#### 1.3 Mensajes
- SystemMessage, HumanMessage, AIMessage
- Conversaciones multi-turno
- Gestión de historial
- Window memory manual

#### 1.4 Estructura de Salida
- StrOutputParser, JsonOutputParser
- PydanticOutputParser
- with_structured_output (nativo Gemini)
- Validación de respuestas
- Caso de uso: extracción legal

### Módulo 2: Memoria ✅

#### 2.1 Memoria a Corto Plazo
- ConversationBufferMemory
- ConversationBufferWindowMemory
- Clase AsistenteConMemoria
- Gestión de historial

#### 2.2 Memoria a Largo Plazo
- ConversationSummaryMemory
- Persistencia en JSON
- Combinación buffer + summary
- Multi-sesión

### Módulo 3: Streaming ✅
- Streaming básico con `.stream()`
- Streaming asíncrono con `.astream()`
- Callbacks para eventos
- Acumulador de tokens
- Streaming con historial
- Comparativa streaming vs batch

### Módulo 4: LangGraph Básico ✅
- ¿Qué es LangGraph y por qué usarlo?
- StateGraph y TypedDict
- Nodes (funciones que procesan estado)
- Edges (conexiones entre nodes)
- Conditional edges (routing dinámico)
- Bucles (loops)
- START y END
- Visualización de grafos

### Módulo 5: Herramientas ✅

#### 5.1 Creación Básica
- Decorador `@tool`
- Propiedades automáticas (nombre, descripción, schema)
- Type hints requeridos
- Docstring como descripción para el LLM
- Errores comunes y mejores prácticas

#### 5.2 Schema Personalizado
- Pydantic para schemas complejos
- `Field` con descripciones
- Validación automática
- `Literal` para valores restringidos
- Listas como argumentos
- Múltiples argumentos con valores por defecto

#### 5.3 ToolNode en LangGraph
- Qué es ToolNode y por qué usarlo
- Ejecución automática de tools
- Paralelismo en ejecución
- Error handling automático
- Estado personalizado con ToolNode
- Grafo básico vs avanzado

### Módulo 6: Human in the Loop ✅

#### 6.1 Introducción a Interrupts
- Qué es `interrupt()` y cómo funciona
- Requisitos: checkpointer, thread ID
- Pausar ejecución y guardar estado
- Reanudar con `Command(resume=valor)`

#### 6.2 Aprobación o Rechazo
- Patrón Approval Workflow
- Routear basado en decisión humana
- Ejemplo: aprobación de emails
- Ejemplo: aprobación de acciones legales

#### 6.3 Revisión y Edición
- Patrón Review and Edit
- Permitir edición de contenido LLM
- Ejemplo: revisión de documentos legales
- Checklists de validación

#### 6.4 Interrupts en Herramientas
- Tools que requieren aprobación
- Editar tool calls antes de ejecutar
- Ejemplo: enviar comunicado oficial
- Integración con ToolNode

#### 6.5 Múltiples Interrupts
- Parallel branches con interrupts
- Manejar múltiples aprobaciones
- Mapa de IDs para resume
- Ejemplo: aprobaciones en paralelo

#### 6.6 Time Travel
- Volver a estados anteriores
- Checkpoint history
- Re-ejecutar desde checkpoint
- Depuración con LangSmith

---

## 🎯 Próximos Módulos a Desarrollar

### Módulo 5: Herramientas (Tools)
- 5.1 Creación de herramientas con @tool
- 5.2 Tool calling con LLMs
- 5.3 Herramientas personalizadas
- 5.4 Multi-tool scenarios
- 5.5 Error handling en tools

### Módulo 6: Human in the Loop
- 6.1 Breakpoints e interrupciones
- 6.2 Human approval
- 6.3 Edición de estado
- 6.4 Time travel (replay)

### Módulo 7: RAG Fundamentos
- 7.1 ¿Qué es RAG y arquitectura?
- 7.2 Document loaders (PDF, TXT, Web)
- 7.3 Chunking strategies
- 7.4 Embeddings con Gemini
- 7.5 Qdrant vector store
- 7.6 Retrieval básico
- 7.7 Reranking con FlashRank

### Módulo 8: RAG Avanzado con LangGraph
- 8.1 Agentic RAG
- 8.2 Fusion retrieval
- 8.3 Parent document retriever
- 8.4 RAG con memoria
- 8.5 Evaluation de RAG

### Módulo 9: Patrones Avanzados
- 9.1 ReAct pattern
- 9.2 Multi-agent systems
- 9.3 Reflection pattern
- 9.4 Supervisor architecture

### Módulo 10: Producción
- 10.1 Checkpointers y persistencia
- 10.2 Caching
- 10.3 LangSmith tracing
- 10.4 Performance tuning

### Módulo 11: Proyecto Final
- 11.1 RAG Legal completo
- 11.2 Integración de todos los componentes
- 11.3 Testing end-to-end
- 11.4 Deployment con Docker

---

## 📝 Tareas Pendientes

### Documentación
- [ ] Completar Módulo 5 (Herramientas)
- [ ] Completar Módulo 6 (Human in the Loop)
- [ ] Completar Módulo 7 (RAG Fundamentos)
- [ ] Completar Módulo 8 (RAG Avanzado)
- [ ] Completar Módulo 9 (Patrones Avanzados)
- [ ] Completar Módulo 10 (Producción)
- [ ] Completar Módulo 11 (Proyecto Final)

### Código
- [ ] Crear ejemplos para Módulo 2.2 (persistencia)
- [ ] Crear ejemplos para Módulo 5
- [ ] Crear ejemplos para Módulo 6
- [ ] Crear ejemplos para Módulo 7
- [ ] Crear ejemplos para Módulo 8
- [ ] Crear ejemplos para Módulo 9
- [ ] Crear ejemplos para Módulo 10
- [ ] Crear proyecto final integrado

### Mejoras
- [ ] Agregar tests unitarios a cada módulo
- [ ] Crear notebook Jupyter para módulos interactivos
- [ ] Agregar diagramas Mermaid en cada lección
- [ ] Crear archivo de ejercicios con soluciones
- [ ] Agregar glosario de términos

---

## 🔗 Recursos Útiles

### Documentación Oficial
- [LangChain Docs](https://docs.langchain.com/oss/python/langchain)
- [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph)
- [Google Gemini API](https://ai.google.dev/docs)

### Repositorios
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)

### Comunidades
- [LangChain Discord](https://discord.gg/langchain)
- [Stack Overflow - langchain](https://stackoverflow.com/questions/tagged/langchain)

---

## 📞 Cómo Usar Este Curso

1. **Orden Recomendado**: Sigue los módulos en orden (0 → 11)
2. **Teoría + Práctica**: Lee la documentación y luego ejecuta el código
3. **Ejercicios**: Completa los ejercicios al final de cada lección
4. **Ritmo**: Dedica 1-2 horas por módulo
5. **Dudas**: Revisa la documentación oficial o pregunta en foros

---

*Documento de progreso actualizado: 2026-03-29*
