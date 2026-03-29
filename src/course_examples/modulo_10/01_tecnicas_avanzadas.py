"""
Módulo 10 - RAG Avanzado: Técnicas y Métodos

Objetivo: Implementar técnicas avanzadas de RAG
Basado en: Análisis de 31 scripts de all_rag_techniques_runnable_scripts
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno
load_dotenv()


# ============================================================================
# EJEMPLO 1: FUSION RETRIEVAL (HÍBRIDO: VECTOR + BM25)
# ============================================================================

def ejemplo_fusion_retrieval():
    """
    Ejemplo 1: Fusion Retrieval - Combina búsqueda vectorial y BM25.
    
    Ventajas:
    - Vector search: Captura significado semántico
    - BM25: Captura keywords exactas
    - Combinación: Mejor de ambos mundos
    """
    
    print("=" * 80)
    print("EJEMPLO 1: FUSION RETRIEVAL (VECTOR + BM25)")
    print("=" * 80)
    
    try:
        from langchain_core.documents import Document
        from rank_bm25 import BM25Okapi
        import numpy as np
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        # 1. Crear documentos de ejemplo
        print("\n1️⃣ Crear documentos:")
        
        documentos = [
            Document(page_content="La acción de tutela protege derechos fundamentales en Colombia", metadata={"id": 1}),
            Document(page_content="El derecho de petición permite solicitar información a autoridades", metadata={"id": 2}),
            Document(page_content="Habeas Corpus protege la libertad personal contra detenciones ilegales", metadata={"id": 3}),
            Document(page_content="La tutela procede cuando no hay otro medio de defensa judicial", metadata={"id": 4}),
            Document(page_content="Derechos fundamentales incluyen vida, salud, educación y debido proceso", metadata={"id": 5}),
        ]
        
        print(f"   Documentos: {len(documentos)}")
        
        # 2. Crear embeddings y vector store
        print("\n2️⃣ Crear vector store:")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        client = QdrantClient(":memory:")
        vector_store = QdrantVectorStore.from_documents(
            documents=documentos,
            embedding=embeddings,
            client=client,
            collection_name="fusion_test"
        )
        
        print(f"   ✅ Vector store creado")
        
        # 3. Crear índice BM25
        print("\n3️⃣ Crear índice BM25:")
        
        tokenized_docs = [doc.page_content.split() for doc in documentos]
        bm25 = BM25Okapi(tokenized_docs)
        
        print(f"   ✅ BM25 index creado")
        print(f"   Vocabulary size: {len(bm25.idf)}")
        
        # 4. Función de fusion retrieval
        print("\n4️⃣ Fusion Retrieval:")
        
        def fusion_retrieval(query: str, k: int = 3, alpha: float = 0.5):
            """
            Fusion retrieval combinando vector y BM25.
            
            Args:
                query: Query de búsqueda
                k: Número de documentos a retornar
                alpha: Peso para vector search (0-1)
            """
            
            # Obtener todos los documentos
            all_docs = vector_store.similarity_search("", k=len(documentos))
            
            # BM25 scores
            bm25_scores = bm25.get_scores(query.split())
            
            # Vector scores
            vector_results = vector_store.similarity_search_with_score(query, k=len(all_docs))
            vector_scores = np.array([score for _, score in vector_results])
            
            # Normalizar (Min-Max)
            vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
            
            # Combinar
            combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
            
            # Top-K
            sorted_indices = np.argsort(combined_scores)[::-1]
            top_docs = [all_docs[i] for i in sorted_indices[:k]]
            
            return top_docs, combined_scores[sorted_indices[:k]]
        
        # 5. Probar con diferentes queries
        print("\n5️⃣ Probar retrieval:")
        
        queries = [
            "tutela derechos",
            "petición información",
            "libertad personal"
        ]
        
        for query in queries:
            print(f"\n   Query: '{query}'")
            
            # Vector search puro
            vector_docs = vector_store.similarity_search(query, k=2)
            print(f"   Vector search: {[doc.metadata['id'] for doc in vector_docs]}")
            
            # BM25 puro
            bm25_scores = bm25.get_scores(query.split())
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:2]
            print(f"   BM25 search: {[documentos[i].metadata['id'] for i in top_bm25_idx]}")
            
            # Fusion
            fusion_docs, fusion_scores = fusion_retrieval(query, k=2, alpha=0.5)
            print(f"   Fusion (α=0.5): {[doc.metadata['id'] for doc in fusion_docs]}")
            print(f"   Scores: {[f'{s:.3f}' for s in fusion_scores]}")
        
        print("\n✅ Fusion Retrieval completado")
        print("   💡 Tip: Ajusta alpha para balancear vector vs keyword search")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   ⚠️ Asegúrate de tener installadas las dependencias:")
        print("      pip install rank-bm25 numpy")


# ============================================================================
# EJEMPLO 2: CONTEXTUAL COMPRESSION
# ============================================================================

def ejemplo_contextual_compression():
    """
    Ejemplo 2: Contextual Compression - Comprimir documentos para extraer solo lo relevante.
    
    Flujo:
    1. Retrieval inicial (top-10)
    2. LLM extrae solo lo relevante para la query
    3. Resultado: documentos más cortos y relevantes
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 2: CONTEXTUAL COMPRESSION")
    print("=" * 80)
    
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.documents import Document
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        # 1. Crear documentos largos
        print("\n1️⃣ Crear documentos largos:")
        
        documentos = [
            Document(page_content="""
            ACCIÓN DE TUTELA - Artículo 86 Constitución Política
            
            La acción de tutela es un mecanismo constitucional de protección inmediata de derechos fundamentales.
            Procede cuando el afectado no disponga de otro medio de defensa judicial, salvo que se utilice como mecanismo transitorio para evitar un perjuicio irremediable.
            
            La tutela procede contra acciones u omisiones de autoridades públicas o de particulares en los casos señalados en la ley.
            
            El juez de tutela ordenará hacer cesar la amenaza o violación del derecho fundamental y restablecer el derecho conculcado.
            
            El fallo de tutela es de inmediato cumplimiento y puede ser impugnado ante el juez superior.
            """, metadata={"source": "constitucion"}),
            
            Document(page_content="""
            DERECHO DE PETICIÓN - Artículo 23 Constitución Política
            
            Toda persona tiene derecho a presentar peticiones respetuosas a las autoridades por motivos de interés general o particular y a obtener pronta resolución.
            
            El legislador podrá reglamentar su ejercicio ante organizaciones privadas para garantizar los derechos fundamentales.
            
            No se requiere abogado para ejercer el derecho de petición.
            
            El silencio administrativo negativo opera cuando la autoridad no responde dentro del término legal (15 días hábiles generalmente).
            """, metadata={"source": "constitucion"}),
            
            Document(page_content="""
            HABEAS CORPUS - Artículo 30 Constitución Política
            
            Quien estuviere privado de su libertad, y creyere estarlo ilegalmente, tiene derecho a invocar ante cualquier autoridad judicial, por sí o por interpuesta persona, el Habeas Corpus.
            
            El Habeas Corpus debe resolverse en un término no mayor a treinta y seis horas.
            
            La detención preventiva no puede exceder el tiempo fijado por la ley.
            """, metadata={"source": "constitucion"})
        ]
        
        total_chars = sum(len(doc.page_content) for doc in documentos)
        print(f"   Documentos: {len(documentos)}")
        print(f"   Total caracteres: {total_chars}")
        
        # 2. Crear vector store
        print("\n2️⃣ Crear vector store:")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        client = QdrantClient(":memory:")
        vector_store = QdrantVectorStore.from_documents(
            documents=documentos,
            embedding=embeddings,
            client=client,
            collection_name="compression_test"
        )
        
        # 3. Crear retriever base
        print("\n3️⃣ Crear retriever base:")
        
        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": 2}
        )
        
        print(f"   ✅ Retriever configurado (k=2)")
        
        # 4. Crear LLM para compresión
        print("\n4️⃣ Crear extractor con LLM:")
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        
        extractor = LLMChainExtractor.from_llm(llm)
        
        print(f"   ✅ Extractor creado")
        
        # 5. Crear compression retriever
        print("\n5️⃣ Crear compression retriever:")
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=extractor,
            base_retriever=base_retriever
        )
        
        print(f"   ✅ Compression retriever listo")
        
        # 6. Probar retrieval con y sin compresión
        print("\n6️⃣ Probar retrieval:")
        
        query = "¿Cuánto tiempo tienen para responder una tutela?"
        
        # Sin compresión
        print(f"\n   Query: '{query}'")
        print(f"\n   📄 SIN compresión:")
        docs_base = base_retriever.invoke(query)
        
        for i, doc in enumerate(docs_base, 1):
            print(f"   Doc {i}: {len(doc.page_content)} caracteres")
            print(f"   Preview: {doc.page_content[:100]}...")
        
        # Con compresión
        print(f"\n   📄 CON compresión:")
        docs_compressed = compression_retriever.invoke(query)
        
        for i, doc in enumerate(docs_compressed, 1):
            print(f"   Doc {i}: {len(doc.page_content)} caracteres")
            print(f"   Preview: {doc.page_content[:100]}...")
        
        # Calcular ahorro
        original_chars = sum(len(doc.page_content) for doc in docs_base)
        compressed_chars = sum(len(doc.page_content) for doc in docs_compressed)
        ahorro = ((original_chars - compressed_chars) / original_chars) * 100
        
        print(f"\n   📊 Ahorro: {ahorro:.1f}% menos caracteres")
        print(f"   Original: {original_chars} chars → Comprimido: {compressed_chars} chars")
        
        print("\n✅ Contextual Compression completado")
        print("   💡 Tip: Usar cuando los documentos son largos y costosos")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


# ============================================================================
# EJEMPLO 3: HyDe (HYPOTHETICAL DOCUMENT EMBEDDING)
# ============================================================================

def ejemplo_hyde():
    """
    Ejemplo 3: HyDe - Generar documento hipotético y buscar similares.
    
    Idea: En vez de buscar con la query directa, generar una respuesta hipotética
    y buscar documentos similares a esa respuesta hipotética.
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 3: HyDe (HYPOTHETICAL DOCUMENT EMBEDDING)")
    print("=" * 80)
    
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from langchain_core.documents import Document
        
        # 1. Crear documentos
        print("\n1️⃣ Crear documentos:")
        
        documentos = [
            Document(page_content="La tutela protege derechos fundamentales como vida, salud, educación, debido proceso", metadata={"id": 1}),
            Document(page_content="El derecho de petición permite solicitar información y obtener respuesta en 15 días", metadata={"id": 2}),
            Document(page_content="Habeas Corpus protege contra detenciones ilegales y debe resolverse en 36 horas", metadata={"id": 3}),
        ]
        
        print(f"   Documentos: {len(documentos)}")
        
        # 2. Crear vector store
        print("\n2️⃣ Crear vector store:")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        client = QdrantClient(":memory:")
        vector_store = QdrantVectorStore.from_documents(
            documents=documentos,
            embedding=embeddings,
            client=client,
            collection_name="hyde_test"
        )
        
        print(f"   ✅ Vector store creado")
        
        # 3. Crear HyDe retriever
        print("\n3️⃣ Crear HyDe retriever:")
        
        class HyDERetriever:
            def __init__(self, vectorstore):
                self.vectorstore = vectorstore
                
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    temperature=0.7
                )
                
                self.hyde_prompt = PromptTemplate(
                    input_variables=["query", "chunk_size"],
                    template="""Dada la pregunta '{query}', genera un documento hipotético que responda directamente.
                    El documento debe ser detallado y tener exactamente {chunk_size} caracteres.
                    
                    Documento hipotético:"""
                )
                
                self.chain = self.hyde_prompt | self.llm
            
            def generate_hypothetical_document(self, query: str, chunk_size: int = 200) -> str:
                input_vars = {"query": query, "chunk_size": chunk_size}
                response = self.chain.invoke(input_vars)
                return response.content
            
            def retrieve(self, query: str, k: int = 2) -> tuple:
                hypothetical_doc = self.generate_hypothetical_document(query)
                similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
                return similar_docs, hypothetical_doc
        
        hyde_retriever = HyDERetriever(vector_store)
        
        print(f"   ✅ HyDe retriever creado")
        
        # 4. Probar HyDe vs retrieval normal
        print("\n4️⃣ Probar HyDe vs Normal:")
        
        query = "¿Qué mecanismo protege la libertad personal?"
        
        # Retrieval normal
        print(f"\n   Query: '{query}'")
        print(f"\n   📄 Retrieval NORMAL:")
        normal_docs = vector_store.similarity_search(query, k=2)
        
        for i, doc in enumerate(normal_docs, 1):
            print(f"   Doc {i} (ID={doc.metadata['id']}): {doc.page_content[:100]}...")
        
        # HyDe retrieval
        print(f"\n   📄 Retrieval HyDe:")
        hyde_docs, hypothetical_doc = hyde_retriever.retrieve(query, k=2)
        
        print(f"\n   Documento HIPOTÉTICO generado:")
        print(f"   {hypothetical_doc[:200]}...")
        
        for i, doc in enumerate(hyde_docs, 1):
            print(f"   Doc {i} (ID={doc.metadata['id']}): {doc.page_content[:100]}...")
        
        print("\n✅ HyDe completado")
        print("   💡 Tip: HyDe ayuda cuando hay brecha semántica entre query y docs")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


# ============================================================================
# EJEMPLO 4: QUERY TRANSFORMATIONS
# ============================================================================

def ejemplo_query_transformations():
    """
    Ejemplo 4: Query Transformations - Transformar queries para mejorar retrieval.
    
    Técnicas:
    1. Query Rewriting (reescritura más específica)
    2. Step-Back Query (query más general para contexto)
    3. Sub-Query Decomposition (descomponer en queries simples)
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 4: QUERY TRANSFORMATIONS")
    print("=" * 80)
    
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # 1. Crear transformer
        print("\n1️⃣ Crear Query Transformer:")
        
        class QueryTransformer:
            def __init__(self):
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
            
            def rewrite_query(self, original_query: str) -> str:
                """Reescribir query para hacerla más específica."""
                
                prompt = PromptTemplate(
                    input_variables=["original_query"],
                    template="""Reescribe la siguiente query para mejorar la recuperación de información.
                    Hazla más específica, detallada y usa términos técnicos apropiados.
                    
                    Query original: {original_query}
                    
                    Query reescrita:"""
                )
                
                chain = prompt | self.llm
                response = chain.invoke({"original_query": original_query})
                
                return response.content.strip()
            
            def generate_step_back_query(self, original_query: str) -> str:
                """Generar query más general para obtener contexto amplio."""
                
                prompt = PromptTemplate(
                    input_variables=["original_query"],
                    template="""Genera una query más general (step-back) que pueda obtener información de contexto relevante.
                    La query debe ser más amplia y conceptual.
                    
                    Query original: {original_query}
                    
                    Step-back query:"""
                )
                
                chain = prompt | self.llm
                response = chain.invoke({"original_query": original_query})
                
                return response.content.strip()
            
            def decompose_query(self, original_query: str) -> list[str]:
                """Descomponer query compleja en sub-queries simples."""
                
                prompt = PromptTemplate(
                    input_variables=["original_query"],
                    template="""Descompón la query en 2-4 sub-queries más simples que, cuando se respondan juntas,
                    den una respuesta completa a la query original.
                    
                    Query original: {original_query}
                    
                    Sub-queries (una por línea, numeradas):"""
                )
                
                chain = prompt | self.llm
                response = chain.invoke({"original_query": original_query})
                
                # Parsear respuesta
                lines = response.content.strip().split('\n')
                sub_queries = []
                
                for line in lines:
                    line = line.strip()
                    if line and line[0].isdigit():
                        # Remover número y punto
                        query = line.split('.', 1)[-1].strip()
                        sub_queries.append(query)
                
                return sub_queries
        
        transformer = QueryTransformer()
        
        print(f"   ✅ Query Transformer creado")
        
        # 2. Probar transformaciones
        print("\n2️⃣ Probar transformaciones:")
        
        queries = [
            "¿Qué es una tutela?",
            "¿Cómo funciona el derecho de petición?",
            "¿Cuáles son los requisitos para un habeas corpus?"
        ]
        
        for original_query in queries:
            print(f"\n{'='*60}")
            print(f"Query Original: '{original_query}'")
            print(f"{'='*60}")
            
            # 1. Query Rewriting
            rewritten = transformer.rewrite_query(original_query)
            print(f"\n📝 Rewritten Query:")
            print(f"   {rewritten}")
            
            # 2. Step-Back Query
            step_back = transformer.generate_step_back_query(original_query)
            print(f"\n🔙 Step-Back Query:")
            print(f"   {step_back}")
            
            # 3. Sub-Query Decomposition
            sub_queries = transformer.decompose_query(original_query)
            print(f"\n🔻 Sub-Queries:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"   {i}. {sq}")
        
        print("\n\n✅ Query Transformations completado")
        print("   💡 Tip: Usar cuando las queries son vagas o complejas")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todos los ejemplos del Módulo 10.
    """
    
    print("=" * 80)
    print("MÓDULO 10: RAG AVANZADO - TÉCNICAS Y MÉTODOS")
    print("=" * 80)
    print("\nEste módulo cubre técnicas avanzadas de RAG:")
    print("1. Fusion Retrieval (Vector + BM25)")
    print("2. Contextual Compression")
    print("3. HyDe (Hypothetical Document Embedding)")
    print("4. Query Transformations")
    print("\nTécnicas adicionales documentadas:")
    print("5. CRAG (Corrective RAG)")
    print("6. Self-RAG")
    print("7. RAPTOR (Recursive Trees)")
    print("8. Graph RAG")
    
    # Ejecutar ejemplos
    ejemplo_fusion_retrieval()
    ejemplo_contextual_compression()
    ejemplo_hyde()
    ejemplo_query_transformations()
    
    print("\n" + "=" * 80)
    print("MÓDULO 10 COMPLETADO")
    print("=" * 80)
    print("\n📚 Próximos pasos:")
    print("   - Módulo 11: Patrones Avanzados de Agentes")
    print("   - Módulo 12: Producción y Deploy")
    print("\n💡 Tips:")
    print("   - Fusion retrieval: Mejor precisión en la mayoría de casos")
    print("   - Compression: Ahorra tokens cuando docs son largos")
    print("   - HyDe: Útil para queries complejas")
    print("   - Query transformations: Mejora retrieval sin cambiar embeddings")


if __name__ == "__main__":
    main()
