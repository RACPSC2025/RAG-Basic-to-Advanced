import streamlit as st
import time
import os
import sys

# Agregar el directorio raíz al path para que pueda importar 'proyectos'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from proyectos.Rag_Legal.graph import get_graph
from proyectos.Rag_Legal.state import RagState
from proyectos.Rag_Legal.ingestor import ingest_pdf

st.set_page_config(
    page_title="Analista Juridico",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Analista Juridico")
st.markdown("""
**Asistente Jurídico Autónomo.** Desarrollado con LangGraph, implementa patrones avanzados: 
*CRAG* (filtrado de documentos irrelevantes) y *Self-RAG* (verificación anti-alucinaciones).
Potenciado por **AWS Bedrock** (Amazon Nova Lite + Titan Embeddings v2).
""")

# INGESTA EN LA BARRA LATERAL
with st.sidebar:
    st.header("🏢 Base de Conocimiento")
    st.info("La base de datos vectorial Chroma se comparte desde `storage/`.")
    
    uploaded_file = st.file_uploader("Ingestar nuevo PDF (Normativa/Decreto)", type=["pdf"])
    if uploaded_file and st.button("Procesar e Indexar"):
        with st.spinner("Procesando PDF con AWS Bedrock (Titan Embeddings v2)..."):
            # Guardar en tmp temporal
            temp_path = os.path.join(os.environ.get("TEMP", "/tmp"), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                st.info("⏳ Conectando a AWS Bedrock...")
                stats = ingest_pdf(temp_path)
                st.success(
                    f"✅ ¡Ingesta completa con AWS Bedrock!\n"
                    f"- Chunks nuevos: {stats['new_chunks_added']}\n"
                    f"- Duplicados omitidos: {stats['duplicates_skipped']}\n"
                    f"- Colección: `{stats.get('collection', 'rag_legal_bedrock')}`"
                )
            except Exception as e:
                st.error(f"❌ Error en la ingesta: {e}")
                st.info("Verifica que las credenciales AWS en .env sean correctas y que la región us-east-2 tenga acceso a Bedrock.")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# CHAT INTERFAZ FUNCIONAL
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            with st.expander("Ver Detalles de Evaluación y Fuentes"):
                if msg["metadata"].get("source_docs"):
                    st.markdown("**Fuentes utilizadas:**")
                    for doc in msg["metadata"]["source_docs"]:
                        st.caption(f"- {doc}")
                st.markdown(f"**Score Grillo (Utilidad):** {msg['metadata'].get('grade')}")
                st.markdown(f"**Score Alucinación:** {msg['metadata'].get('hallucination_score', 0.0)}")
                st.markdown(f"**Intentos de Generación:** {msg['metadata'].get('attempts', 0)}")

# Input del usuario
prompt = st.chat_input("Escriba su consulta legal sobre la documentación indexada...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando marco legal y orquestando agente evaluador..."):
            try:
                graph = get_graph()
                
                # Ejecutar el grafo de estado
                estado_inicial = RagState(question=prompt)
                
                # Para mostrar estado intermedio podríamos usar graph.stream, 
                # pero invoke es más sencillo para la primera versión
                final_state = graph.invoke(estado_inicial)
                
                respuesta = final_state["generation"]
                
                metadata = {
                    "source_docs": final_state.get("source_docs", []),
                    "grade": final_state.get("grade", "Desconocido"),
                    "hallucination_score": final_state.get("hallucination_score", 0.0),
                    "attempts": final_state.get("attempts", 0)
                }

                # Evalución visual del veredicto del agente
                if final_state.get("attempts", 0) > 1:
                    st.warning("⚠️ El agente detectó una alucinación y tuvo que reescribir la respuesta basándose solo en la evidencia.")
                
            except Exception as e:
                respuesta = f"Hubo un error de conexión con la API o base de datos: {e}"
                metadata = {}

        st.markdown(respuesta)
        if metadata.get("source_docs"):
            with st.expander("Ver Detalles de Evaluación y Fuentes"):
                st.markdown("**Fuentes utilizadas:**")
                for doc in metadata["source_docs"]:
                    st.caption(f"- {doc}")
                st.markdown(f"**Score de Utilidad (CRAG):** {metadata.get('grade')}")
                st.markdown(f"**Score Alucinación:** {metadata.get('hallucination_score')}")
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": respuesta,
        "metadata": metadata
    })
