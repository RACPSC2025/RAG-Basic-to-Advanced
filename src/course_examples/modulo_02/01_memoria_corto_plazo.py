"""
01_memoria_corto_plazo.py
Memoria a Corto Plazo con LangChain

Objetivo: Dominar ConversationBufferMemory y WindowMemory
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
)
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
)


# ============================================================
# 1. CONVERSATION BUFFER MEMORY (BÁSICO)
# ============================================================

def conversation_buffer_memory():
    """Memoria que guarda TODA la conversación"""
    
    print("=" * 60)
    print("CONVERSATION BUFFER MEMORY")
    print("=" * 60)
    
    # Crear memoria
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )
    
    # Crear prompt con placeholder para historial
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Simular conversación
    print("\n--- Conversación ---")
    
    # Turno 1
    input_usuario = "Hola, me llamo Carlos"
    messages = prompt.format_messages(
        chat_history=memory.load_memory_variables({})["chat_history"],
        input=input_usuario
    )
    response = llm.invoke(messages)
    print(f"U: {input_usuario}")
    print(f"AI: {response.content}")
    
    # Guardar en memoria
    memory.save_context(
        {"input": input_usuario},
        {"output": response.content}
    )
    
    # Turno 2
    input_usuario = "¿Cuál es mi nombre?"
    messages = prompt.format_messages(
        chat_history=memory.load_memory_variables({})["chat_history"],
        input=input_usuario
    )
    response = llm.invoke(messages)
    print(f"\nU: {input_usuario}")
    print(f"AI: {response.content}")
    
    # Ver historial completo
    print(f"\n--- Historial en Memoria ---")
    historial = memory.load_memory_variables({})
    print(f"Mensajes guardados: {len(historial['chat_history'])}")
    
    for msg in historial["chat_history"]:
        print(f"  {type(msg).__name__}: {msg.content[:50]}...")


# ============================================================
# 2. CONVERSATION BUFFER WINDOW MEMORY
# ============================================================

def conversation_window_memory():
    """Memoria que guarda solo los últimos N mensajes"""
    
    print("\n" + "=" * 60)
    print("CONVERSATION BUFFER WINDOW MEMORY")
    print("=" * 60)
    
    # Crear memoria con ventana de 2 turnos
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=2,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Simular conversación larga
    print("\n--- Conversación Larga (k=2) ---")
    
    conversaciones = [
        "Hola, ¿qué tal?",
        "Bien, gracias. ¿Te gusta el fútbol?",
        "Sí, me encanta. ¿Y a ti?",
        "También me gusta. ¿Qué equipo prefieres?",
        "Prefiero el Real Madrid. ¿Y tú?",
        "Yo prefiero el Barcelona.",
        "¿Quién ganó el último clásico?",
    ]
    
    for input_usuario in conversaciones:
        messages = prompt.format_messages(
            chat_history=memory.load_memory_variables({})["chat_history"],
            input=input_usuario
        )
        response = llm.invoke(messages)
        
        print(f"U: {input_usuario}")
        print(f"AI: {response.content[:80]}...")
        print()
        
        memory.save_context(
            {"input": input_usuario},
            {"output": response.content}
        )
    
    # Ver qué quedó en memoria
    print("--- Lo que quedó en memoria (últimos 2 turnos) ---")
    historial = memory.load_memory_variables({})
    print(f"Mensajes guardados: {len(historial['chat_history'])}")
    
    for msg in historial["chat_history"]:
        print(f"  {type(msg).__name__}: {msg.content}")


# ============================================================
# 3. CLASE REUTILIZABLE CON MEMORIA
# ============================================================

class AsistenteConMemoria:
    """Asistente con memoria integrada"""
    
    def __init__(self, system_prompt: str = "Eres un asistente útil.", k: int = 5):
        self.system_prompt = system_prompt
        
        # Memoria con ventana
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=k,
        )
        
        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        # Chain
        self.chain = self.prompt | llm
    
    def preguntar(self, input_usuario: str) -> str:
        """Hacer una pregunta y obtener respuesta"""
        
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=input_usuario
        )
        
        response = self.chain.invoke(messages)
        
        self.memory.save_context(
            {"input": input_usuario},
            {"output": response.content}
        )
        
        return response.content
    
    def limpiar_memoria(self):
        """Limpiar toda la memoria"""
        self.memory.clear()
        print("✅ Memoria limpiada")
    
    def ver_historial(self):
        """Ver el historial actual"""
        historial = self.memory.load_memory_variables({})["chat_history"]
        
        if not historial:
            print("📭 Sin historial")
            return
        
        print(f"📜 Historial ({len(historial)} mensajes):")
        for msg in historial:
            emoji = "🤖" if isinstance(msg, AIMessage) else "👤"
            print(f"  {emoji} {msg.content[:60]}...")


def usar_asistente_con_memoria():
    """Demostrar la clase AsistenteConMemoria"""
    
    print("=" * 60)
    print("ASISTENTE CON MEMORIA")
    print("=" * 60)
    
    asistente = AsistenteConMemoria(
        system_prompt="Eres un asistente legal experto en derecho colombiano.",
        k=3
    )
    
    # Turno 1
    print("\n--- Turno 1 ---")
    respuesta = asistente.preguntar("¿Qué es una acción de tutela?")
    print(f"R: {respuesta[:100]}...")
    
    # Turno 2
    print("\n--- Turno 2 ---")
    respuesta = asistente.preguntar("¿Quién puede interponerla?")
    print(f"R: {respuesta[:100]}...")
    
    # Turno 3
    print("\n--- Turno 3 ---")
    respuesta = asistente.preguntar("¿Cuánto tiempo tiene el juez?")
    print(f"R: {respuesta[:100]}...")
    
    # Turno 4
    print("\n--- Turno 4 ---")
    respuesta = asistente.preguntar("¿Y si no responde?")
    print(f"R: {respuesta[:100]}...")
    
    # Ver historial
    print("\n--- Historial ---")
    asistente.ver_historial()
    
    # Turno 5
    print("\n--- Turno 5 ---")
    respuesta = asistente.preguntar("¿De qué hablábamos al principio?")
    print(f"R: {respuesta[:100]}...")
    
    # Limpiar memoria
    asistente.limpiar_memoria()


if __name__ == "__main__":
    conversation_buffer_memory()
    conversation_window_memory()
    usar_asistente_con_memoria()
