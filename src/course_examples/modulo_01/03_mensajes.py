"""
03_mensajes.py
Trabajando con Mensajes y Chat Models

Objetivo: Dominar el manejo de conversaciones con historial
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
)


# ============================================================
# 1. MENSAJES BÁSICOS
# ============================================================

def mensajes_basicos():
    """Crear y usar mensajes individuales"""
    
    # Forma 1: Usando clases directamente
    system_msg = SystemMessage(content="Eres un asistente útil.")
    human_msg = HumanMessage(content="¿Hola, cómo estás?")
    
    print("=" * 60)
    print("MENSAJES BÁSICOS")
    print("=" * 60)
    
    response = llm.invoke([system_msg, human_msg])
    print(f"System: {system_msg.content}")
    print(f"Human: {human_msg.content}")
    print(f"AI: {response.content}\n")
    
    # Forma 2: Usando tuplas (más conciso)
    messages = [
        ("system", "Eres un asistente útil."),
        ("user", "¿Hola, cómo estás?"),
    ]
    
    response = llm.invoke(messages)
    print(f"Con tuplas: {response.content}\n")


# ============================================================
# 2. CONVERSACIÓN MULTI-TURNO
# ============================================================

def conversacion_multi_turno():
    """Mantener una conversación con historial"""
    
    print("=" * 60)
    print("CONVERSACIÓN MULTI-TURNO")
    print("=" * 60)
    
    # Historial de la conversación
    conversation_history = [
        ("system", "Eres un asistente legal colombiano."),
        
        # Turno 1
        ("user", "¿Qué es una tutela?"),
        ("assistant", "La acción de tutela es un mecanismo constitucional para proteger derechos fundamentales."),
        
        # Turno 2
        ("user", "¿Quién puede interponerla?"),
        ("assistant", "Cualquier persona puede interponer una tutela, no requiere abogado."),
        
        # Turno 3 (nueva pregunta)
        ("user", "¿Cuánto tiempo tiene el juez para responder?"),
    ]
    
    response = llm.invoke(conversation_history)
    
    print(f"Última pregunta: ¿Cuánto tiempo tiene el juez para responder?")
    print(f"Respuesta: {response.content}\n")


# ============================================================
# 3. CLASES DE MENSAJES EXPLÍCITAS
# ============================================================

def clases_explicitas():
    """Usar las clases de mensajes explícitamente"""
    
    print("=" * 60)
    print("CLASES EXPLÍCITAS DE MENSAJES")
    print("=" * 60)
    
    # Crear mensajes con clases explícitas
    messages = [
        SystemMessage(content="Eres un profesor de derecho."),
        HumanMessage(content="Explícame qué es el derecho penal"),
        AIMessage(content="El derecho penal es la rama del derecho que define los delitos y las penas."),
        HumanMessage(content="¿Y el derecho civil?"),
    ]
    
    response = llm.invoke(messages)
    
    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content[:60]}...")
    
    print(f"\nAI Response: {response.content}\n")


# ============================================================
# 4. GESTIÓN MANUAL DEL HISTORIAL
# ============================================================

class ConversacionLegal:
    """Clase para gestionar una conversación legal"""
    
    def __init__(self, especialidad: str = "colombiano"):
        self.especialidad = especialidad
        self.historial = [
            SystemMessage(
                content=f"Eres un asistente legal experto en derecho {especialidad}."
            )
        ]
    
    def preguntar(self, pregunta: str) -> str:
        """Hacer una pregunta y obtener respuesta"""
        
        # Agregar pregunta al historial
        self.historial.append(HumanMessage(content=pregunta))
        
        # Obtener respuesta
        response = llm.invoke(self.historial)
        
        # Agregar respuesta al historial
        self.historial.append(AIMessage(content=response.content))
        
        return response.content
    
    def obtener_historial(self) -> list:
        """Obtener todo el historial"""
        return self.historial
    
    def limpiar_historial(self):
        """Limpiar el historial manteniendo el system prompt"""
        self.historial = self.historial[:1]  # Mantener solo system message


def usar_clase_conversacion():
    """Demostrar la clase ConversacionLegal"""
    
    print("=" * 60)
    print("CLASE CONVERSACION LEGAL")
    print("=" * 60)
    
    conversacion = ConversacionLegal(especialidad="laboral colombiano")
    
    # Turno 1
    print("\n--- Turno 1 ---")
    respuesta = conversacion.preguntar("¿Cuántos días de vacaciones me corresponden?")
    print(f"R: {respuesta[:150]}...")
    
    # Turno 2 (con contexto)
    print("\n--- Turno 2 ---")
    respuesta = conversacion.preguntar("¿Y si gané más del mínimo?")
    print(f"R: {respuesta[:150]}...")
    
    # Turno 3
    print("\n--- Turno 3 ---")
    respuesta = conversacion.preguntar("¿Cómo se calculan las cesantías?")
    print(f"R: {respuesta[:150]}...")
    
    # Ver historial
    print("\n--- Historial Completo ---")
    historial = conversacion.obtener_historial()
    print(f"Total mensajes: {len(historial)}")


# ============================================================
# 5. LIMITAR HISTORIAL (WINDOW)
# ============================================================

def limitar_historial():
    """Mantener solo los últimos N mensajes"""
    
    print("=" * 60)
    print("LIMITAR HISTORIAL (WINDOW)")
    print("=" * 60)
    
    # Simular conversación larga
    historial_completo = [
        SystemMessage(content="Eres un asistente útil."),
    ]
    
    # Agregar 10 turnos
    for i in range(10):
        historial_completo.append(HumanMessage(content=f"Pregunta {i+1}"))
        historial_completo.append(AIMessage(content=f"Respuesta {i+1}"))
    
    print(f"Historial completo: {len(historial_completo)} mensajes")
    
    # Mantener solo últimos 4 mensajes (2 turnos)
    window_size = 4
    historial_limitado = historial_completo[-window_size:]
    
    print(f"Historial limitado: {len(historial_limitado)} mensajes")
    print(f"Últimos mensajes:")
    for msg in historial_limitado:
        print(f"  - {type(msg).__name__}: {msg.content}")


if __name__ == "__main__":
    mensajes_basicos()
    conversacion_multi_turno()
    clases_explicitas()
    usar_clase_conversacion()
    limitar_historial()
