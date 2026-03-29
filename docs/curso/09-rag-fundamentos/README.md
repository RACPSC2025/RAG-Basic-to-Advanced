# Módulo 9: RAG Fundamentos

> **Basado en**: [Documentación Oficial de LangChain - RAG](https://docs.langchain.com/oss/python/langchain/rag)  
> **Estado**: ✅ En Desarrollo  
> **Prerrequisitos**: Módulos 1-6 completados

---

## 📋 Índice del Módulo

1. [9.1 - ¿Qué es RAG y Por Qué Usarlo?](#91-qué-es-rag-y-por-qué-usarlo)
2. [9.2 - Document Loaders (Carga de Documentos)](#92-document-loaders-carga-de-documentos)
3. [9.3 - Chunking Strategies (Segmentación)](#93-chunking-strategies-segmentación)
4. [9.4 - Embeddings con Google Gemini](#94-embeddings-con-google-gemini)
5. [9.5 - Vector Stores (Qdrant)](#95-vector-stores-qdrant)
6. [9.6 - Retrieval Básico](#96-retrieval-básico)
7. [9.7 - Reranking con FlashRank](#97-reranking-con-flashrank)

---

## 9.1 - ¿Qué es RAG y Por Qué Usarlo?

### ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** es una técnica que combina:
- **Retrieval**: Búsqueda de información relevante en una base de conocimientos
- **Generation**: Generación de respuestas usando un LLM

```
┌─────────────────────────────────────────────────────────┐
│                    ARQUITECTURA RAG                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. INDEXING (Pre-proceso)                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Documentos│→│  Split   │→│ Embeddings │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                      ↓                                  │
│  ┌────────────────────────────────────┐                 │
│  │         Vector Store (Qdrant)      │                 │
│  └────────────────────────────────────┘                 │
│                                                         │
│  2. QUERY TIME (Runtime)                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Query   │→│ Retrieval │→│  LLM +   │              │
│  │ Usuario  │  │  (Top-K) │  │ Contexto │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                      ↓                                  │
│  ┌────────────────────────────────────┐                 │
│  │         Respuesta Generada         │                 │
│  └────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

### ¿Por Qué Usar RAG?

| Problema | Solución RAG |
|----------|--------------|
| **LLM sin conocimiento específico** | RAG provee contexto relevante |
| **Información desactualizada** | Vector store se puede actualizar |
| **Alucinaciones del LLM** | Respuestas basadas en documentos reales |
| **Falta de transparencia** | Se puede mostrar las fuentes |
| **Costo de fine-tuning** | Más barato que fine-tuning |

### Casos de Uso Típicos

```python
# ✅ Casos de uso ideales para RAG:

# 1. Q&A sobre documentos internos
#    - Manuales de empresa
#    - Políticas internas
#    - Documentación técnica

# 2. Asistentes legales
#    - Búsqueda de jurisprudencia
#    - Consultas sobre leyes
#    - Análisis de contratos

# 3. Soporte técnico
#    - Base de conocimientos
#    - Troubleshooting guides
#    - FAQs

# 4. Investigación
#    - Papers académicos
#    - Artículos científicos
#    - Datos estructurados
```

---

## 9.2 - Document Loaders (Carga de Documentos)

### ¿Qué son Document Loaders?

Los **Document Loaders** cargan documentos de diversas fuentes y los convierten al formato `Document` de LangChain.

```python
from langchain_core.documents import Document

# Formato estándar
document = Document(
    page_content="Contenido del documento...",
    metadata={
        "source": "archivo.pdf",
        "page": 1,
        "author": "Autor"
    }
)
```

### Loaders para PDFs

```python
from langchain_community.document_loaders import PyPDFLoader

# Cargar PDF
loader = PyPDFLoader("documento.pdf")
documents = loader.load()

print(f"Páginas cargadas: {len(documents)}")
print(f"Contenido página 1: {documents[0].page_content[:200]}")
```

### Loaders para Múltiples Formatos

```python
from langchain_community.document_loaders import (
    PyPDFLoader,      # PDFs
    TextLoader,       # TXT
    UnstructuredWordDocumentLoader,  # DOCX
    CSVLoader,        # CSV
    WebBaseLoader      # Webs
)

# PDF
pdf_loader = PyPDFLoader("documento.pdf")
pdf_docs = pdf_loader.load()

# Texto
txt_loader = TextLoader("documento.txt")
txt_docs = txt_loader.load()

# Web
web_loader = WebBaseLoader("https://ejemplo.com")
web_docs = web_loader.load()
```

### Cargar Múltiples Archivos

```python
from langchain_community.document_loaders import DirectoryLoader

# Cargar todos los PDFs de un directorio
loader = DirectoryLoader(
    "docs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()
print(f"Total documentos: {len(documents)}")
```

---

## 9.3 - Chunking Strategies (Segmentación)

### ¿Por Qué Hacer Chunking?

Los documentos grandes deben dividirse en chunks más pequeños porque:
1. **Context window limitada** del LLM
2. **Retrieval más preciso** (chunks específicos)
3. **Mejor uso de tokens** (solo lo relevante)

### RecursiveCharacterTextSplitter (Recomendado)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración recomendada
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Tamaño máximo por chunk
    chunk_overlap=200,    # Overlap entre chunks
    length_function=len,  # Función para medir longitud
    separators=[          # Separadores en orden recursivo
        "\n\n",  # Primero por párrafos
        "\n",    # Luego por líneas
        " ",     # Luego por palabras
        ""       # Finalmente por caracteres
    ]
)

# Usar el splitter
chunks = text_splitter.split_documents(documents)
print(f"Chunks creados: {len(chunks)}")
```

### Parámetros Clave

```python
# chunk_size: Tamaño máximo del chunk
chunk_size=500    # Chunks pequeños (más precisos)
chunk_size=1000   # Balanceado (RECOMENDADO)
chunk_size=2000   # Chunks grandes (más contexto)

# chunk_overlap: Overlap entre chunks
chunk_overlap=0     # Sin overlap (puede perder contexto)
chunk_overlap=200   # Overlap moderado (RECOMENDADO)
chunk_overlap=500   # Mucho overlap (más tokens)

# separators: Orden de separación
separators=["\n\n", "\n", " ", ""]  # Estándar
separators=["。", "、", ""]          # Para chino/japonés
```

### CharacterTextSplitter (Alternativa Simple)

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
```

---

## 9.4 - Embeddings con Google Gemini

### ¿Qué son Embeddings?

Los **embeddings** son representaciones vectoriales de texto que capturan significado semántico.

```
"tutela" → [0.1, -0.5, 0.3, ..., 0.8]  # Vector de 768 dimensiones
"derecho" → [0.2, -0.4, 0.2, ..., 0.7]

# Textos similares tienen vectores similares
cosine_similarity(embedding("tutela"), embedding("derecho")) = 0.85
```

### GoogleGenerativeAIEmbeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Inicializar embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Modelo de Gemini
    task_type="retrieval_document",  # Para documentos
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Crear embedding de un texto
texto = "La acción de tutela protege derechos fundamentales"
vector = embeddings.embed_query(texto)

print(f"Dimensión del vector: {len(vector)}")  # 768
```

### Embeddings para Documents vs Queries

```python
# Para documentos (indexing)
embeddings_doc = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document"
)

# Para queries (búsqueda)
embeddings_query = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_query"
)

# Usar el embedding correcto mejora la precisión
```

---

## 9.5 - Vector Stores (Qdrant)

### ¿Qué es un Vector Store?

Un **Vector Store** almacena embeddings y permite búsqueda por similitud.

```
┌─────────────────────────────────────────────────────────┐
│                 VECTOR STORE (Qdrant)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Documento 1 → Embedding 1 → Vector Store               │
│  Documento 2 → Embedding 2 → Vector Store               │
│  Documento 3 → Embedding 3 → Vector Store               │
│                                                         │
│  Query → Embedding Query → Búsqueda Similitud           │
│              ↓                                          │
│  Top-K Vectores Más Similares → Documentos              │
└─────────────────────────────────────────────────────────┘
```

### Qdrant Local (Desarrollo)

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Cliente local (en memoria)
client = QdrantClient(":memory:")

# Crear vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="documentos_legales",
    embedding=embeddings
)

# Agregar documentos
vector_store.add_documents(documents=chunks)
```

### Qdrant con Persistencia

```python
from qdrant_client import QdrantClient

# Cliente con persistencia en disco
client = QdrantClient(
    path="./qdrant_storage",  # Directorio local
    port=6333
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="documentos_legales",
    embedding=embeddings
)
```

### Qdrant Cloud (Producción)

```python
from qdrant_client import QdrantClient

# Cliente cloud
client = QdrantClient(
    url="https://xxx-xxx.us-east.aws.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="documentos_legales",
    embedding=embeddings
)
```

---

## 9.6 - Retrieval Básico

### Similarity Search

```python
# Búsqueda básica por similitud
query = "¿Qué es una acción de tutela?"

# Top-K documentos más similares
docs = vector_store.similarity_search(
    query=query,
    k=3  # Top 3 resultados
)

for doc in docs:
    print(f"Documento: {doc.page_content[:200]}")
    print(f"Metadata: {doc.metadata}\n")
```

### Similarity Search con Score

```python
# Búsqueda con scores de similitud
results = vector_store.similarity_search_with_score(
    query=query,
    k=3
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Contenido: {doc.page_content[:200]}\n")
```

### Crear Retriever

```python
# Convertir vector store a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",  # o "mmr"
    search_kwargs={"k": 3}
)

# Usar retriever
docs = retriever.invoke(query)
```

### MMR (Maximal Marginal Relevance)

```python
# MMR: Balancea relevancia y diversidad
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,  # Traer 10, seleccionar 3
        "lambda_mult": 0.5  # 0=diversidad, 1=relevancia
    }
)

docs = retriever.invoke(query)
```

---

## 9.7 - Reranking con FlashRank

### ¿Por Qué Reranking?

El reranking mejora los resultados del retrieval:
1. **Retrieval inicial**: Trae documentos relevantes (rápido)
2. **Reranking**: Reordena por relevancia (más preciso)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankReranker
from langchain_google_genai import ChatGoogleGenerativeAI

# Crear reranker
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
reranker = FlashrankReranker(llm=llm, top_n=3)

# Crear retriever con compresión
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

# Usar
docs = compression_retriever.invoke(query)
# Trae 10, rerankea, retorna top 3
```

### FlashRankReranker

```python
from langchain.retrievers.document_compressors import FlashrankReranker

# Reranker básico
reranker = FlashrankReranker(
    model="ms-marco-MiniLM-L-12-v2",  # Modelo pre-entrenado
    top_n=3  # Top N resultados
)

# Usar con documentos
docs = reranker.compress_documents(
    query=query,
    documents=retrieved_docs
)
```

---

## 🎯 Ejercicio Práctico: RAG Básico

### Implementación Completa

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# 1. Cargar documentos
print("1️⃣ Cargando documentos...")
loader = PyPDFLoader("documento_legal.pdf")
documents = loader.load()

# 2. Segmentar documentos
print("2️⃣ Segmentando documentos...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"   Chunks creados: {len(chunks)}")

# 3. Crear embeddings
print("3️⃣ Creando embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document"
)

# 4. Almacenar en Qdrant
print("4️⃣ Almacenando en Qdrant...")
client = QdrantClient(":memory:")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,
    collection_name="documentos_legales"
)

# 5. Crear retriever
print("5️⃣ Creando retriever...")
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 6. Probar retrieval
print("6️⃣ Probando retrieval...")
query = "¿Qué es una acción de tutela?"
docs = retriever.invoke(query)

print(f"\n📚 Documentos encontrados: {len(docs)}\n")
for i, doc in enumerate(docs, 1):
    print(f"Documento {i}:")
    print(f"  Contenido: {doc.page_content[:200]}...")
    print(f"  Metadata: {doc.metadata}\n")

print("✅ RAG básico completado exitosamente!")
```

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)
- [Text Splitters](https://docs.langchain.com/oss/python/integrations/splitters)
- [Qdrant Vector Store](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)
- [Google Embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/google_genai)

### Siguiente Lección
➡️ **Módulo 10: RAG Avanzado**

---

*Módulo creado: 2026-03-29*  
*Basado en documentación oficial de LangChain (Mayo 2025)*
