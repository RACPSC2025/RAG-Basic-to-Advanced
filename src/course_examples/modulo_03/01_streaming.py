"""
01_streaming.py
Streaming de Tokens con LangChain

Objetivo: Dominar streaming en tiempo real
"""

import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
    streaming=True,
)


# ============================================================
# 1. STREAMING BÁSICO
# ============================================================

def streaming_basico():
    """Streaming básico token por token"""
    
    print("=" * 60)
    print("STREAMING BÁSICO")
    print("=" * 60)
    print("\nRespuesta streaming:\n")
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Explícame qué es una acción de tutela en 3 frases")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    start_time = time.time()
    
    for token in chain.stream({}):
        print(token, end="", flush=True)
        time.sleep(0.02)
    
    elapsed = time.time() - start_time
    print(f"\n\n⏱️ Tiempo total: {elapsed:.2f} segundos")


# ============================================================
# 2. STREAMING CON ACUMULADOR
# ============================================================

def streaming_con_acumulador():
    """Acumular tokens mientras se hace streaming"""
    
    print("\n" + "=" * 60)
    print("STREAMING CON ACUMULADOR")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Dame 5 consejos para estudiar derecho")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    respuesta_completa = ""
    
    print("\nGenerando respuesta:\n")
    
    for token in chain.stream({}):
        print(token, end="", flush=True)
        respuesta_completa += token
        time.sleep(0.02)
    
    print(f"\n\n✅ Respuesta acumulada ({len(respuesta_completa)} caracteres)")
    print(f"📄 Primeras 100 letras: {respuesta_completa[:100]}...")


# ============================================================
# 3. STREAMING EN CHAT CON HISTORIAL
# ============================================================

def streaming_con_historial():
    """Streaming en conversación con memoria"""
    
    print("\n" + "=" * 60)
    print("STREAMING CON HISTORIAL")
    print("=" * 60)
    
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.prompts import MessagesPlaceholder
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente legal colombiano."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Turno 1
    print("\n--- Turno 1 ---")
    input_usuario = "¿Qué es el derecho de petición?"
    
    messages = prompt.format_messages(
        chat_history=memory.load_memory_variables({})["chat_history"],
        input=input_usuario
    )
    
    respuesta = ""
    for token in llm.stream(messages):
        print(token.content, end="", flush=True)
        respuesta += token.content
        time.sleep(0.02)
    
    memory.save_context(
        {"input": input_usuario},
        {"output": respuesta}
    )
    
    # Turno 2
    print("\n\n--- Turno 2 (con contexto) ---")
    input_usuario = "¿Cuánto tiempo tienen para responder?"
    
    messages = prompt.format_messages(
        chat_history=memory.load_memory_variables({})["chat_history"],
        input=input_usuario
    )
    
    respuesta = ""
    for token in llm.stream(messages):
        print(token.content, end="", flush=True)
        respuesta += token.content
        time.sleep(0.02)


# ============================================================
# 4. CLASE CON STREAMING
# ============================================================

class AsistenteStreaming:
    """Asistente con soporte para streaming"""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.memory = []
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ])
        
        self.chain = self.prompt | llm | StrOutputParser()
    
    def preguntar(self, input_usuario: str, usar_streaming: bool = True) -> str:
        """Preguntar con opción de streaming"""
        
        messages = self.prompt.format_messages(
            chat_history=self.memory,
            input=input_usuario
        )
        
        if usar_streaming:
            print("\n🤖 AI: ", end="", flush=True)
            
            respuesta = ""
            for token in self.chain.stream({"chat_history": self.memory, "input": input_usuario}):
                print(token, end="", flush=True)
                respuesta += token
                time.sleep(0.01)
            
            print()
        else:
            respuesta = self.chain.invoke({
                "chat_history": self.memory,
                "input": input_usuario
            })
            print(f"\n🤖 AI: {respuesta}")
        
        self.memory.append(HumanMessage(content=input_usuario))
        self.memory.append(AIMessage(content=respuesta))
        
        return respuesta
    
    def limpiar(self):
        """Limpiar memoria"""
        self.memory = []
        print("✅ Memoria limpiada")


def usar_asistente_streaming():
    """Demostrar asistente con streaming"""
    
    print("=" * 60)
    print("ASISTENTE CON STREAMING")
    print("=" * 60)
    
    asistente = AsistenteStreaming("Eres un profesor de derecho.")
    
    asistente.preguntar("¿Qué es la Constitución?", usar_streaming=True)
    asistente.preguntar("¿Cuál es la Constitución vigente en Colombia?", usar_streaming=True)
    asistente.preguntar("¿Cuántos artículos tiene?", usar_streaming=False)


# ============================================================
# 5. COMPARACIÓN: STREAMING VS NO STREAMING
# ============================================================

def comparar_streaming_vs_no_streaming():
    """Comparar experiencia con y sin streaming"""
    
    print("=" * 60)
    print("COMPARACIÓN: STREAMING vs NO STREAMING")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Explica los 3 poderes del estado en Colombia")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # SIN STREAMING
    print("\n--- SIN STREAMING ---")
    start = time.time()
    respuesta = chain.invoke({})
    elapsed = time.time() - start
    
    print(f"⏳ Esperaste {elapsed:.2f} segundos sin ver nada")
    print(f"✅ Respuesta de {len(respuesta)} caracteres")
    
    # CON STREAMING
    print("\n--- CON STREAMING ---")
    start = time.time()
    
    respuesta_stream = ""
    for token in chain.stream({}):
        print(token, end="", flush=True)
        respuesta_stream += token
        time.sleep(0.02)
    
    elapsed = time.time() - start
    
    print(f"\n⏱️ Tiempo total: {elapsed:.2f} segundos")
    print(f"✅ Usuario vio progreso desde el inicio")
    
    print(f"\n📊 COMPARACIÓN:")
    print(f"   Sin streaming: {len(respuesta)} chars (espera total)")
    print(f"   Con streaming: {len(respuesta_stream)} chars (progreso visible)")


if __name__ == "__main__":
    streaming_basico()
    streaming_con_acumulador()
    streaming_con_historial()
    usar_asistente_streaming()
    comparar_streaming_vs_no_streaming()
