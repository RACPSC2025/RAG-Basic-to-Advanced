# ⚖️ RAG MVP Legal - Proyecto Fénix

Este repositorio consolida nuestro Minimum Viable Product (MVP) para procesar, indexar y recuperar información precisa sobre documentación legal y normativa en Colombia, apoyándose en la base de datos vectorial Chroma existente y el motor de inferencia de Gemini.

## 🏛️ Caso de Uso Principal
El sistema procesa decretos, directivas presidenciales (e.g. *Departamento Administrativo de la Función Pública, Directivas Presidenciales de 2024*) y normativas de obligatorio cumplimiento.
Dada la criticidad del dominio (Legal), las alucinaciones del modelo son inaceptables.

---

## 🚀 Plan de Acción - Aether (Senior AI Architect)

Basado en las mejores prácticas de **Agentic RAG**, **Corrective RAG (CRAG)** y **Self-RAG** (orquestados por *LangGraph*), ejecutaremos la implementación en 4 fases escalonadas:

### ⚙️ Fase 1: Core de Inteligencia y Persistencia (Setup)
- [ ] **Configuración Segura (`proyectos/Rag_Legal/config/`)**:
  - Iniciar clase con inyección de dependencias para manejar la API gratuita de **Google Gemini** de forma óptima (`google-genai`).
  - Setear `ChatGoogleGenerativeAI` para la generación.
  - Setear `GoogleGenerativeAIEmbeddings` para respetar la compatibilidad con el índice previamente creado en `storage/chroma.sqlite3`.
- [ ] **Capa de Persistencia Vectorial**:
  - Conectar el `Chroma Retriever` asegurando apuntar al directorio absolute `storage/`.

### 🔍 Fase 2: Retrieval Avanzado y Re-ranking (Opcional según viabilidad free)
- [ ] Preparar módulos para *Semantic Search* estricto (alto índice de similitud mínimo por defecto) para que el retriever descarte todo lo irrelevante.

### 🧠 Fase 3: Orquestación del Agente Legal (LangGraph)
Aquí usaremos un patrón **CRAG + Self-RAG** implementando nodos específicos:
- [ ] **Node 1 - `retrieve_documents`**: Busca en nuestro ChromaDB.
- [ ] **Node 2 - `grade_documents` (CRAG)**: Un LLM evalúa si cada chunk devuelto de verdad contiene fragmentos que respondan a la consulta. Filtra los irrelevantes.
- [ ] **Node 3 - `generate_response` (Self-RAG)**: Genera la respuesta estrictamente basándose en los documentos filtrados. Debe citar capítulos/artículos exactos.
- [ ] **Node 4 - `check_hallucinations`**: Un pase final donde el LLM (como juez imparcial) contrasta su propia respuesta contra los fragmentos de entrada. Si halla una discrepancia, fuerza al grafo a re-intentar o contestar *"No dispongo de esa información"*.

### 💻 Fase 4: Punto de Entrada (User Interface)
- [ ] Exponer una interfaz, ya sea mediante CLI (`app.py`) o un pequeño orquestador de Streamlit, para poder realizar todo el flujo transaccional.

---

*Diseñado y planificado por **Aether** - Advanced Agentic Coding.*
