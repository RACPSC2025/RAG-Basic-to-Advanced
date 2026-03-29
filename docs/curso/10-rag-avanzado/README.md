# Módulo 10: RAG Avanzado - Técnicas y Métodos

> **Basado en**: Análisis de 31 scripts de `all_rag_techniques_runnable_scripts`  
> **Estado**: ✅ En Desarrollo  
> **Prerrequisitos**: Módulo 9 (RAG Fundamentos) completado

---

## 📋 Índice del Módulo

1. [10.1 - Fusion Retrieval (Híbrido)](#101-fusion-retrieval-híbrido)
2. [10.2 - Contextual Compression](#102-contextual-compression)
3. [10.3 - HyDe (Hypothetical Document Embedding)](#103-hyde-hypothetical-document-embedding)
4. [10.4 - Query Transformations](#104-query-transformations)
5. [10.5 - CRAG (Corrective RAG)](#105-crag-corrective-rag)
6. [10.6 - Self-RAG](#106-self-rag)
7. [10.7 - RAPTOR (Recursive Trees)](#107-raptor-recursive-trees)
8. [10.8 - Graph RAG](#108-graph-rag)

---

## 10.1 - Fusion Retrieval (Híbrido)

### ¿Qué es Fusion Retrieval?

**Fusion Retrieval** combina dos técnicas de búsqueda:
- **Vector Search**: Búsqueda semántica (embeddings)
- **BM25**: Búsqueda por palabras clave

```
┌─────────────────────────────────────────────────────────┐
│                 FUSION RETRIEVAL                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query → ┌─────────────┐                               │
│          │  Split      │                               │
│          └──────┬──────┘                               │
│                 ↓                                      │
│    ┌────────────┴────────────┐                         │
│    ↓                         ↓                         │
│ ┌──────────┐           ┌──────────┐                   │
│ │  Vector  │           │   BM25   │                   │
│ │  Search  │           │  Search  │                   │
│ │ (Dense)  │           │ (Sparse) │                   │
│ └────┬─────┘           └────┬─────┘                   │
│      ↓                      ↓                         │
│ ┌──────────────────────────────┐                      │
│ │    Normalizar Scores         │                      │
│ │  (Min-Max Normalization)     │                      │
│ └────────────┬─────────────────┘                      │
│              ↓                                        │
│ ┌──────────────────────────────┐                      │
│ │  Combinar: α*vector + (1-α)*bm25 │                 │
│ └────────────┬─────────────────┘                      │
│              ↓                                        │
│         Top-K Documentos                              │
└─────────────────────────────────────────────────────────┘
```

### Implementación

```python
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

def create_bm25_index(documents: list[Document]) -> BM25Okapi:
    """Crear índice BM25 para búsqueda por palabras clave."""
    
    # Tokenizar documentos
    tokenized_docs = [doc.page_content.split() for doc in documents]
    
    # Crear índice BM25
    bm25 = BM25Okapi(tokenized_docs)
    
    return bm25

def fusion_retrieval(
    vectorstore,
    bm25: BM25Okapi,
    query: str,
    k: int = 5,
    alpha: float = 0.5
) -> list[Document]:
    """
    Fusion retrieval combinando vector search y BM25.
    
    Args:
        vectorstore: Vector store para búsqueda semántica
        bm25: Índice BM25 para búsqueda por keywords
        query: Query de búsqueda
        k: Número de documentos a retornar
        alpha: Peso para vector search (1-alpha para BM25)
    
    Returns:
        Top-K documentos combinados
    """
    
    # 1. Obtener todos los documentos
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    
    # 2. Búsqueda BM25
    bm25_scores = bm25.get_scores(query.split())
    
    # 3. Búsqueda Vectorial
    vector_results = vectorstore.similarity_search_with_score(
        query, 
        k=len(all_docs)
    )
    
    # 4. Normalizar scores (Min-Max)
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    
    # 5. Combinar scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    
    # 6. Ordenar y retornar top-K
    sorted_indices = np.argsort(combined_scores)[::-1]
    
    return [all_docs[i] for i in sorted_indices[:k]]
```

### Parámetros Clave

```python
# alpha: Balance entre vector y keyword search
alpha = 0.5  # Balanceado (RECOMENDADO)
alpha = 0.7  # Más peso a vector search
alpha = 0.3  # Más peso a keyword search

# k: Número de documentos
k = 5   # Para respuestas concisas
k = 10  # Para contexto amplio
```

---

## 10.2 - Contextual Compression

### ¿Qué es Contextual Compression?

**Contextual Compression** comprime los documentos recuperados para extraer solo la información relevante:

```
┌─────────────────────────────────────────────────────────┐
│            CONTEXTUAL COMPRESSION                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query → Retrieval → Top-10 Documentos                  │
│                      ↓                                  │
│           ┌─────────────────────┐                       │
│           │  LLM Chain          │                       │
│           │  Extractor          │                       │
│           │                     │                       │
│           │ "Extrae solo lo     │                       │
│           │  relevante para     │                       │
│           │  la query"          │                       │
│           └────────┬────────────┘                       │
│                    ↓                                    │
│           Top-10 → Top-3 (comprimidos)                  │
│                    ↓                                    │
│              LLM + Contexto                             │
└─────────────────────────────────────────────────────────┘
```

### Implementación

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Crear retriever base
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 2. Crear LLM para compresión
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# 3. Crear extractor
extractor = LLMChainExtractor.from_llm(llm)

# 4. Crear retriever con compresión
compression_retriever = ContextualCompressionRetriever(
    base_compressor=extractor,
    base_retriever=base_retriever
)

# 5. Usar
docs = compression_retriever.invoke(query)

# Resultado: Documentos más cortos, solo información relevante
```

### Ventajas

| Sin Compresión | Con Compresión |
|----------------|----------------|
| 10 docs × 1000 tokens = 10,000 tokens | 3 docs × 300 tokens = 900 tokens |
| Más ruido | Más señal |
| Más costo | Menos costo |

---

## 10.3 - HyDe (Hypothetical Document Embedding)

### ¿Qué es HyDe?

**HyDe** genera un documento hipotético que responde la query, luego busca documentos similares a ese hipotético.

```
┌─────────────────────────────────────────────────────────┐
│                    HyDe RAG                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query: "¿Qué es una tutela?"                           │
│           ↓                                             │
│  ┌─────────────────────────────────┐                   │
│  │  LLM Genera Documento Hipotético │                  │
│  │                                  │                  │
│  │  "La tutela es un mecanismo     │                  │
│  │   constitucional para proteger  │                  │
│  │   derechos fundamentales..."    │                  │
│  └────────────┬────────────────────┘                  │
│               ↓                                        │
│  Embedding del Documento Hipotético                    │
│               ↓                                        │
│  Búsqueda por similitud en Vector Store                │
│               ↓                                        │
│  Documentos reales similares al hipotético             │
└─────────────────────────────────────────────────────────┘
```

### Implementación

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class HyDERetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        
        # LLM para generar documento hipotético
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.7
        )
        
        # Prompt para HyDe
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Dada la pregunta '{query}', genera un documento hipotético que responda directamente.
            El documento debe ser detallado y tener exactamente {chunk_size} caracteres.
            
            Documento hipotético:"""
        )
        
        self.chain = self.hyde_prompt | self.llm
    
    def generate_hypothetical_document(self, query: str, chunk_size: int = 500) -> str:
        """Generar documento hipotético."""
        
        input_vars = {"query": query, "chunk_size": chunk_size}
        response = self.chain.invoke(input_vars)
        
        return response.content
    
    def retrieve(self, query: str, k: int = 3) -> tuple:
        """Retrieval usando documento hipotético."""
        
        # 1. Generar documento hipotético
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # 2. Buscar documentos similares al hipotético
        similar_docs = self.vectorstore.similarity_search(
            hypothetical_doc, 
            k=k
        )
        
        return similar_docs, hypothetical_doc
```

### Cuándo Usar HyDe

✅ **Usar HyDe cuando**:
- Queries son preguntas complejas
- Brecha semántica entre query y documentos
- Documentos técnicos/especializados

❌ **No usar HyDe cuando**:
- Queries son simples búsquedas de keywords
- Se necesita velocidad máxima (HyDe es más lento)

---

## 10.4 - Query Transformations

### Tipos de Transformación

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class QueryTransformer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    # 1. Query Rewriting (Reescritura)
    def rewrite_query(self, original_query: str) -> str:
        """
        Reescribir query para mejorar retrieval.
        Más específica, detallada.
        """
        
        prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""Reescribe la siguiente query para mejorar la recuperación de información.
            Hazla más específica y detallada.
            
            Query original: {original_query}
            
            Query reescrita:"""
        )
        
        chain = prompt | self.llm
        response = chain.invoke({"original_query": original_query})
        
        return response.content
    
    # 2. Step-Back Query (Query más general)
    def generate_step_back_query(self, original_query: str) -> str:
        """
        Generar query más general para obtener contexto amplio.
        """
        
        prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""Genera una query más general que pueda obtener información de contexto relevante.
            
            Query original: {original_query}
            
            Step-back query:"""
        )
        
        chain = prompt | self.llm
        response = chain.invoke({"original_query": original_query})
        
        return response.content
    
    # 3. Sub-Query Decomposition (Descomposición)
    def decompose_query(self, original_query: str) -> list[str]:
        """
        Descomponer query compleja en sub-queries simples.
        """
        
        prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""Descompón la query en 2-4 sub-queries más simples.
            
            Query original: {original_query}
            
            Sub-queries:
            1."""
        )
        
        chain = prompt | self.llm
        response = chain.invoke({"original_query": original_query})
        
        # Parsear respuesta
        sub_queries = [
            q.strip() 
            for q in response.content.split('\n') 
            if q.strip() and q.strip()[0].isdigit()
        ]
        
        return sub_queries
```

### Ejemplo de Uso

```python
transformer = QueryTransformer()

query_original = "¿Cuáles son los impactos del cambio climático en la agricultura colombiana?"

# 1. Query reescrita
rewritten = transformer.rewrite_query(query_original)
# → "Impactos del cambio climático en cultivos de café, arroz y maíz en Colombia"

# 2. Step-back query
step_back = transformer.generate_step_back_query(query_original)
# → "¿Cómo afecta el cambio climático a la agricultura?"

# 3. Sub-queries
sub_queries = transformer.decompose_query(query_original)
# → [
#      "¿Cómo afecta el cambio climático al rendimiento de cultivos?",
#      "¿Qué regiones agrícolas de Colombia son más vulnerables?",
#      "¿Qué medidas de adaptación existen?"
#    ]
```

---

## 10.5 - CRAG (Corrective RAG)

### ¿Qué es CRAG?

**CRAG** evalúa la relevancia de los documentos recuperados y decide:
- ✅ **Correcto**: Usar documentos recuperados
- ❌ **Incorrecto**: Hacer web search
- ⚠️ **Ambiguo**: Combinar ambos

```
┌─────────────────────────────────────────────────────────┐
│                    CRAG RAG                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query → Retrieval → Documentos                         │
│                      ↓                                  │
│           ┌──────────────────┐                          │
│           │  Evaluación      │                          │
│           │  (0-1 score)     │                          │
│           └────────┬─────────┘                          │
│                    ↓                                    │
│      ┌─────────────┼─────────────┐                      │
│      ↓             ↓             ↓                      │
│   Score>0.7    0.3<Score<0.7   Score<0.3               │
│      ↓             ↓             ↓                      │
│  ✅ CORRECTO   ⚠️ AMBIGUO   ❌ INCORRECTO              │
│      ↓             ↓             ↓                      │
│  Usar docs    Combinar docs   Web Search               │
│  recuperados  + Web Search   → docs                    │
│      ↓             ↓             ↓                      │
│           ┌────────┴─────────┐                          │
│           │  Knowledge       │                          │
│           │  Refinement      │                          │
│           └────────┬─────────┘                          │
│                    ↓                                    │
│              Generar Respuesta                          │
└─────────────────────────────────────────────────────────┘
```

### Implementación

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class RetrievalEvaluatorInput(BaseModel):
    """Modelo para evaluar relevancia de documento."""
    relevance_score: float = Field(
        ..., 
        description="Score de relevancia entre 0 y 1"
    )

class CRAGRAG:
    def __init__(self, vectorstore, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        
        # Umbrales
        self.upper_threshold = 0.7
        self.lower_threshold = 0.3
    
    def retrieve_documents(self, query: str, k: int = 3) -> list[str]:
        """Recuperar documentos del vector store."""
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def evaluate_documents(self, query: str, documents: list[str]) -> list[float]:
        """Evaluar relevancia de cada documento."""
        
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""En una escala de 0 a 1, ¿qué tan relevante es el siguiente documento para la query?
            
            Query: {query}
            Documento: {document}
            
            Score de relevancia:"""
        )
        
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        
        scores = []
        for doc in documents:
            result = chain.invoke({"query": query, "document": doc})
            scores.append(result.relevance_score)
        
        return scores
    
    def knowledge_refinement(self, document: str) -> list[str]:
        """Extraer puntos clave de un documento."""
        
        prompt = PromptTemplate(
            input_variables=["document"],
            template="""Extrae la información clave del siguiente documento en formato de puntos:
            
            Documento: {document}
            
            Puntos clave:"""
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"document": document})
        
        return [p.strip() for p in result.content.split('\n') if p.strip()]
    
    def run(self, query: str) -> str:
        """Ejecutar CRAG completo."""
        
        # 1. Retrieval
        retrieved_docs = self.retrieve_documents(query)
        
        # 2. Evaluación
        eval_scores = self.evaluate_documents(query, retrieved_docs)
        
        print(f"Scores: {eval_scores}")
        
        # 3. Decidir acción
        max_score = max(eval_scores)
        
        if max_score > self.upper_threshold:
            # ✅ Correcto: usar docs recuperados
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            
        elif max_score < self.lower_threshold:
            # ❌ Incorrecto: web search
            final_knowledge = self.perform_web_search(query)
            
        else:
            # ⚠️ Ambiguo: combinar
            retrieved_knowledge = self.knowledge_refinement(best_doc)
            web_knowledge = self.perform_web_search(query)
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        
        # 4. Generar respuesta
        response_prompt = PromptTemplate(
            input_variables=["query", "knowledge"],
            template="""Basado en el siguiente conocimiento, responde la query.
            
            Query: {query}
            Conocimiento: {knowledge}
            
            Respuesta:"""
        )
        
        chain = response_prompt | self.llm
        response = chain.invoke({"query": query, "knowledge": final_knowledge})
        
        return response.content
```

---

## 10.6 - Self-RAG

### ¿Qué es Self-RAG?

**Self-RAG** es un proceso reflexivo que:
1. Decide si retrieval es necesario
2. Evalúa relevancia de contextos
3. Genera múltiples respuestas
4. Evalúa soporte y utilidad
5. Selecciona la mejor respuesta

```
┌─────────────────────────────────────────────────────────┐
│                    SELF-RAG                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query                                                  │
│    ↓                                                    │
│  ┌──────────────────┐                                   │
│  │ ¿Retrieval       │── No ──→ Generar sin retrieval   │
│  │ necesario?       │                                   │
│  └────────┬─────────┘                                   │
│           ↓ Sí                                          │
│  Retrieval → Top-K Contextos                            │
│           ↓                                             │
│  ┌──────────────────┐                                   │
│  │ ¿Contexto        │── Irrelevante ──→ Descartar      │
│  │ Relevante?       │                                   │
│  └────────┬─────────┘                                   │
│           ↓ Relevante                                   │
│  Generar Respuesta (por cada contexto)                  │
│           ↓                                             │
│  ┌──────────────────┐                                   │
│  │ ¿Soportada por   │── No ──→ Descartar               │
│  │ el contexto?     │                                   │
│  └────────┬─────────┘                                   │
│           ↓ Sí                                          │
│  Evaluar Utilidad (1-5)                                 │
│           ↓                                             │
│  Seleccionar mejor respuesta (mayor utilidad)           │
└─────────────────────────────────────────────────────────┘
```

---

## 10.7 - RAPTOR (Recursive Trees)

### ¿Qué es RAPTOR?

**RAPTOR** crea un árbol de resúmenes recursivos:

```
┌─────────────────────────────────────────────────────────┐
│                    RAPTOR                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Nivel 0: Documentos Originales                         │
│  [Doc1] [Doc2] [Doc3] [Doc4] [Doc5] [Doc6]              │
│     ↓      ↓      ↓      ↓      ↓      ↓                │
│  Embeddings                                             │
│     ↓                                                   │
│  Clustering (Gaussian Mixture Model)                    │
│     ↓                                                   │
│  Nivel 1: Resúmenes de Cluster                          │
│  [Summary1] [Summary2] [Summary3]                       │
│     ↓      ↓      ↓                                     │
│  Embeddings                                             │
│     ↓                                                   │
│  Clustering                                             │
│     ↓                                                   │
│  Nivel 2: Resúmenes de Alto Nivel                       │
│  [SummaryA] [SummaryB]                                  │
│                                                         │
│  Query → Traverse Tree → Respuesta                      │
└─────────────────────────────────────────────────────────┘
```

---

## 10.8 - Graph RAG

### ¿Qué es Graph RAG?

**Graph RAG** construye un grafo de conocimiento conectando documentos por:
- Similitud de embeddings
- Conceptos compartidos
- Entidades nombradas

```
┌─────────────────────────────────────────────────────────┐
│                    Graph RAG                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Documentos → Extraer Conceptos/Entidades               │
│       ↓                                                 │
│  ┌─────────────────────────────────┐                   │
│  │  Grafo de Conocimiento          │                   │
│  │                                 │                   │
│  │  (Doc1)───[comparte:X]───(Doc2) │                   │
│  │    │                         │                     │
│  │ [sim:0.85]              [sim:0.72]                  │
│  │    │                         │                     │
│  │  (Doc3)───[comparte:Y]───(Doc4) │                   │
│  └─────────────────────────────────┘                   │
│                                                         │
│  Query → Traverse Graph → Contexto Enriquecido          │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Ejercicio Práctico: Comparativa de Técnicas

```python
# Ver archivo: src/course_examples/modulo_10/01_tecnicas_avanzadas.py
```

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [Fusion Retrieval](https://docs.langchain.com/oss/python/langchain/fusion_retrieval)
- [Contextual Compression](https://docs.langchain.com/oss/python/langchain/contextual_compression)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [RAPTOR Paper](https://arxiv.org/abs/2401.18059)

### Siguiente Módulo
➡️ **Módulo 11: Patrones Avanzados de Agentes**

---

*Módulo creado: 2026-03-29*  
*Basado en análisis de 31 scripts RAG*
