"""
Módulo 9 - RAG Fundamentos

Objetivo: Implementar un sistema RAG básico completo
Basado en: https://docs.langchain.com/oss/python/langchain/rag
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno
load_dotenv()


# ============================================================================
# EJEMPLO 1: CARGA DE DOCUMENTOS (DOCUMENT LOADERS)
# ============================================================================

def ejemplo_carga_documentos():
    """
    Ejemplo 1: Carga de documentos desde diferentes fuentes.
    
    Muestra cómo usar diferentes document loaders para:
    - PDFs
    - Textos planos
    - Directorios completos
    """
    
    print("=" * 80)
    print("EJEMPLO 1: CARGA DE DOCUMENTOS")
    print("=" * 80)
    
    # 1. Cargar PDF
    print("\n1️⃣ Cargar PDF:")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        # Verificar si existe el PDF de ejemplo
        pdf_path = Path("../data/sample.pdf")
        if pdf_path.exists():
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            print(f"   ✅ PDF cargado exitosamente")
            print(f"   Número de páginas: {len(documents)}")
            print(f"   Contenido página 1: {documents[0].page_content[:200]}...")
        else:
            print(f"   ⚠️ PDF no encontrado: {pdf_path}")
            print(f"   Usando documento de ejemplo...")
            
            # Crear documento de ejemplo
            from langchain_core.documents import Document
            documents = [
                Document(
                    page_content="""
                    ACCIÓN DE TUTELA
                    
                    La acción de tutela es un mecanismo constitucional para proteger
                    derechos fundamentales. Está consagrada en el Artículo 86 de la
                    Constitución Política de Colombia.
                    
                    Procedencia:
                    - Procede cuando no existan otros medios de defensa
                    - Se puede interponer dentro de los 4 meses siguientes
                    - No requiere abogado
                    
                    Derechos que protege:
                    - Derecho a la vida
                    - Derecho a la salud
                    - Derecho al debido proceso
                    - Derecho de petición
                    - Entre otros derechos fundamentales
                    """,
                    metadata={"source": "ejemplo_tutela.txt", "page": 1}
                )
            ]
            print(f"   ✅ Documento de ejemplo creado")
            print(f"   Número de documentos: {len(documents)}")
            
    except Exception as e:
        print(f"   ❌ Error cargando PDF: {e}")
        from langchain_core.documents import Document
        documents = [Document(page_content="Documento de ejemplo", metadata={})]
    
    # 2. Cargar desde directorio
    print("\n2️⃣ Cargar desde directorio:")
    try:
        from langchain_community.document_loaders import DirectoryLoader
        
        # Crear directorio de ejemplo si no existe
        input_dir = Path("../data/input")
        input_dir.mkdir(parents=True, exist_ok=True)
        
        loader = DirectoryLoader(
            str(input_dir),
            glob="**/*.txt",
            show_progress=True
        )
        
        docs = loader.load()
        print(f"   ✅ Documentos cargados: {len(docs)}")
        
    except Exception as e:
        print(f"   ⚠️ Directorio vacío o error: {e}")
    
    return documents


# ============================================================================
# EJEMPLO 2: SEGMENTACIÓN DE DOCUMENTOS (TEXT SPLITTERS)
# ============================================================================

def ejemplo_segmentacion(documentos):
    """
    Ejemplo 2: Segmentación de documentos en chunks.
    
    Muestra cómo usar diferentes text splitters para:
    - RecursiveCharacterTextSplitter (recomendado)
    - CharacterTextSplitter
    - Comparar resultados
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 2: SEGMENTACIÓN DE DOCUMENTOS")
    print("=" * 80)
    
    # 1. RecursiveCharacterTextSplitter
    print("\n1️⃣ RecursiveCharacterTextSplitter (RECOMENDADO):")
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # Tamaño de chunk pequeño para demo
        chunk_overlap=50,     # Overlap para mantener contexto
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_recursive = text_splitter.split_documents(documentos)
    
    print(f"   ✅ Chunks creados: {len(chunks_recursive)}")
    print(f"   Chunk size configurado: 300 caracteres")
    print(f"   Overlap: 50 caracteres")
    
    # Mostrar primer chunk
    if chunks_recursive:
        print(f"\n   Primer chunk:")
        print(f"   ┌────────────────────────────────────────────────────┐")
        print(f"   │ {chunks_recursive[0].page_content[:200]}...")
        print(f"   └────────────────────────────────────────────────────┘")
    
    # 2. CharacterTextSplitter
    print("\n2️⃣ CharacterTextSplitter:")
    
    from langchain_text_splitters import CharacterTextSplitter
    
    text_splitter_char = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks_char = text_splitter_char.split_documents(documentos)
    
    print(f"   ✅ Chunks creados: {len(chunks_char)}")
    
    # 3. Comparación
    print("\n3️⃣ Comparación:")
    print(f"   RecursiveCharacterTextSplitter: {len(chunks_recursive)} chunks")
    print(f"   CharacterTextSplitter: {len(chunks_char)} chunks")
    print(f"   → RecursiveCharacterTextSplitter generalmente produce mejores resultados")
    
    return chunks_recursive


# ============================================================================
# EJEMPLO 3: EMBEDDINGS CON GOOGLE GEMINI
# ============================================================================

def ejemplo_embeddings():
    """
    Ejemplo 3: Creación de embeddings con Google Gemini.
    
    Muestra cómo:
    - Inicializar embeddings
    - Crear embeddings para texto
    - Diferencia entre embeddings para documentos y queries
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 3: EMBEDDINGS CON GOOGLE GEMINI")
    print("=" * 80)
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        # 1. Inicializar embeddings
        print("\n1️⃣ Inicializar embeddings:")
        
        embeddings_doc = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        embeddings_query = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )
        
        print(f"   ✅ Embeddings inicializados")
        print(f"   Modelo: models/embedding-001")
        print(f"   Task type document: retrieval_document")
        print(f"   Task type query: retrieval_query")
        
        # 2. Crear embedding de ejemplo
        print("\n2️⃣ Crear embedding de ejemplo:")
        
        texto_ejemplo = "La acción de tutela protege derechos fundamentales"
        vector = embeddings_doc.embed_query(texto_ejemplo)
        
        print(f"   Texto: '{texto_ejemplo}'")
        print(f"   Dimensión del vector: {len(vector)}")
        print(f"   Primeros 10 valores: {vector[:10]}")
        
        # 3. Comparar embeddings
        print("\n3️⃣ Comparar embeddings (similitud coseno):")
        
        texto1 = "tutela"
        texto2 = "derecho"
        texto3 = "manzana"
        
        vector1 = embeddings_doc.embed_query(texto1)
        vector2 = embeddings_doc.embed_query(texto2)
        vector3 = embeddings_doc.embed_query(texto3)
        
        # Calcular similitud coseno (simplificado)
        import numpy as np
        
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        sim_12 = cosine_similarity(np.array(vector1), np.array(vector2))
        sim_13 = cosine_similarity(np.array(vector1), np.array(vector3))
        
        print(f"   Similitud('{texto1}', '{texto2}'): {sim_12:.4f}")
        print(f"   Similitud('{texto1}', '{texto3}'): {sim_13:.4f}")
        print(f"   → Textos legalmente relacionados tienen mayor similitud")
        
        return embeddings_doc
        
    except Exception as e:
        print(f"   ❌ Error con embeddings: {e}")
        print(f"   ⚠️ Verifica que GOOGLE_API_KEY esté configurada")
        return None


# ============================================================================
# EJEMPLO 4: VECTOR STORE CON QDRANT
# ============================================================================

def ejemplo_vector_store(chunks, embeddings):
    """
    Ejemplo 4: Almacenar documentos en Qdrant.
    
    Muestra cómo:
    - Crear vector store local (en memoria)
    - Agregar documentos
    - Hacer búsquedas
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 4: VECTOR STORE CON QDRANT")
    print("=" * 80)
    
    if embeddings is None:
        print("   ⚠️ No se pueden crear embeddings sin API key")
        return None
    
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        # 1. Crear cliente Qdrant (en memoria para demo)
        print("\n1️⃣ Crear cliente Qdrant:")
        
        client = QdrantClient(":memory:")
        print(f"   ✅ Cliente creado (en memoria)")
        
        # 2. Crear vector store
        print("\n2️⃣ Crear vector store:")
        
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name="documentos_legales"
        )
        
        print(f"   ✅ Vector store creado")
        print(f"   Colección: documentos_legales")
        print(f"   Documentos indexados: {len(chunks)}")
        
        # 3. Búsqueda básica
        print("\n3️⃣ Búsqueda básica:")
        
        query = "¿Qué es una tutela?"
        docs = vector_store.similarity_search(query, k=2)
        
        print(f"   Query: '{query}'")
        print(f"   Documentos encontrados: {len(docs)}")
        
        for i, doc in enumerate(docs, 1):
            print(f"\n   Documento {i}:")
            print(f"   {doc.page_content[:150]}...")
        
        # 4. Búsqueda con scores
        print("\n4️⃣ Búsqueda con scores:")
        
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for doc, score in results:
            print(f"   Score: {score:.4f} - {doc.page_content[:100]}...")
        
        return vector_store
        
    except Exception as e:
        print(f"   ❌ Error con Qdrant: {e}")
        return None


# ============================================================================
# EJEMPLO 5: RETRIEVAL Y RERANKING
# ============================================================================

def ejemplo_retrieval_reranking(vector_store):
    """
    Ejemplo 5: Retrieval y reranking de documentos.
    
    Muestra cómo:
    - Crear retriever
    - Usar diferentes tipos de búsqueda
    - Aplicar reranking con FlashRank
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 5: RETRIEVAL Y RERANKING")
    print("=" * 80)
    
    if vector_store is None:
        print("   ⚠️ No se puede hacer retrieval sin vector store")
        return
    
    # 1. Crear retriever básico
    print("\n1️⃣ Crear retriever básico:")
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    print(f"   ✅ Retriever creado")
    print(f"   Search type: similarity")
    print(f"   Top-K: 3")
    
    # 2. Probar retrieval
    print("\n2️⃣ Probar retrieval:")
    
    query = "¿Cuáles son los requisitos para interponer una tutela?"
    docs = retriever.invoke(query)
    
    print(f"   Query: '{query}'")
    print(f"   Documentos: {len(docs)}")
    
    for i, doc in enumerate(docs, 1):
        print(f"\n   Doc {i}:")
        print(f"   {doc.page_content[:150]}...")
    
    # 3. MMR Retrieival (Maximal Marginal Relevance)
    print("\n3️⃣ MMR Retrieval (diversidad + relevancia):")
    
    retriever_mmr = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 5,
            "lambda_mult": 0.5
        }
    )
    
    docs_mmr = retriever_mmr.invoke(query)
    print(f"   ✅ MMR retrieval: {len(docs_mmr)} documentos")
    
    # 4. Reranking con FlashRank (si está disponible)
    print("\n4️⃣ Reranking con FlashRank:")
    
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import FlashrankReranker
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        
        reranker = FlashrankReranker(
            llm=llm,
            top_n=2
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever
        )
        
        docs_reranked = compression_retriever.invoke(query)
        
        print(f"   ✅ Reranking completado")
        print(f"   Documentos después de reranking: {len(docs_reranked)}")
        
    except Exception as e:
        print(f"   ⚠️ FlashRank no disponible: {e}")
        print(f"   Usando retrieval básico")


# ============================================================================
# EJEMPLO 6: RAG COMPLETO
# ============================================================================

def ejemplo_rag_completo():
    """
    Ejemplo 6: Sistema RAG completo de principio a fin.
    
    Integra todos los componentes:
    1. Carga de documentos
    2. Segmentación
    3. Embeddings
    4. Vector store
    5. Retrieval
    6. Generación con LLM
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 6: SISTEMA RAG COMPLETO")
    print("=" * 80)
    
    # 1. Carga
    print("\n📥 1. Carga de documentos:")
    documentos = ejemplo_carga_documentos()
    
    # 2. Segmentación
    print("\n✂️  2. Segmentación:")
    chunks = ejemplo_segmentacion(documentos)
    
    # 3. Embeddings
    print("\n🔢 3. Embeddings:")
    embeddings = ejemplo_embeddings()
    
    # 4. Vector store
    print("\n🗄️  4. Vector store:")
    vector_store = ejemplo_vector_store(chunks, embeddings)
    
    # 5. Retrieval
    print("\n🔍 5. Retrieval:")
    ejemplo_retrieval_reranking(vector_store)
    
    # 6. Generación con LLM
    print("\n🤖 6. Generación con LLM:")
    
    if vector_store and embeddings:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        
        # Crear retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        
        # Obtener documentos relevantes
        query = "¿Qué derechos protege la tutela?"
        docs = retriever.invoke(query)
        
        # Contexto
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente legal experto en derecho colombiano.
            Responde basándote ÚNICAMENTE en el contexto proporcionado.
            Si la respuesta no está en el contexto, di que no lo sabes.
            
            Contexto:
            {contexto}
            """),
            ("user", "{pregunta}")
        ])
        
        # LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        
        # Chain
        chain = prompt | llm
        
        # Generar respuesta
        messages = prompt.format_messages(
            contexto=contexto,
            pregunta=query
        )
        
        response = llm.invoke(messages)
        
        print(f"   Pregunta: {query}")
        print(f"   Respuesta: {response.content[:300]}...")
    
    print("\n✅ SISTEMA RAG COMPLETADO EXITOSAMENTE!")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todos los ejemplos del Módulo 9.
    """
    
    print("=" * 80)
    print("MÓDULO 9: RAG FUNDAMENTOS")
    print("=" * 80)
    print("\nEste módulo cubre los fundamentos de RAG (Retrieval-Augmented Generation):")
    print("1. Document Loaders")
    print("2. Text Splitters")
    print("3. Embeddings")
    print("4. Vector Stores")
    print("5. Retrieval")
    print("6. Reranking")
    print("7. RAG Completo")
    
    # Ejecutar ejemplo completo
    ejemplo_rag_completo()
    
    print("\n" + "=" * 80)
    print("MÓDULO 9 COMPLETADO")
    print("=" * 80)
    print("\n📚 Próximos pasos:")
    print("   - Módulo 10: RAG Avanzado (Agentic RAG, CRAG, Self-RAG)")
    print("   - Módulo 11: Patrones Avanzados")
    print("   - Módulo 12: Producción")
    print("\n💡 Tips:")
    print("   - Ajusta chunk_size según tu caso de uso")
    print("   - Usa overlap para mantener contexto entre chunks")
    print("   - Experimenta con diferentes k en retrieval")
    print("   - Considera reranking para mejorar precisión")


if __name__ == "__main__":
    main()
