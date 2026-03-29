# Módulo 2.1: Memoria a Corto Plazo

## Objetivos
- Comprender qué es la memoria en LangChain
- Usar ConversationBufferMemory
- Implementar ConversationBufferWindowMemory
- Gestionar el historial de chat automáticamente

---

## 2.1.1 ¿Qué es la Memoria en LangChain?

La **memoria** permite que un agente/LLM recuerde conversaciones previas.

```
┌─────────────────────────────────────────────────────────┐
│              SIN MEMORIA (Stateless)                    │
├─────────────────────────────────────────────────────────┤
│  Usuario: "¿Cuál es mi nombre?"                         │
│  AI: "No lo sé, no me lo has dicho."                    │
│                                                         │
│  Usuario: "Me llamo Juan"                               │
│  AI: "Mucho gusto Juan."                                │
│                                                         │
│  Usuario: "¿Cuál es mi nombre?" ← ¡No recuerda!         │
│  AI: "No lo sé, no me lo has dicho."                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              CON MEMORIA (Stateful)                     │
├─────────────────────────────────────────────────────────┤
│  Usuario: "¿Cuál es mi nombre?"                         │
│  AI: "No lo sé, no me lo has dicho."                    │
│  [Memoria guarda el historial]                          │
│                                                         │
│  Usuario: "Me llamo Juan"                               │
│  AI: "Mucho gusto Juan."                                │
│  [Memoria actualiza historial]                          │
│                                                         │
│  Usuario: "¿Cuál es mi nombre?" ← ¡Sí recuerda!         │
│  AI: "Tu nombre es Juan."                               │
└─────────────────────────────────────────────────────────┘
```

---

## 2.1.2 Código de Ejemplo

Archivo: `src/course_examples/modulo_02/01_memoria_corto_plazo.py`

```python
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
        memory_key="chat_history",  # Nombre de la variable en el prompt
        return_messages=True,       # Retornar como lista de mensajes
        input_key="input",          # Nombre del input del usuario
        output_key="output",        # Nombre del output del asistente
    )
    
    # Crear prompt con placeholder para historial
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil."),
        MessagesPlaceholder(variable_name="chat_history"),  # Aquí va el historial
        ("user", "{input}"),
    ])
    
    # Crear chain
    chain = prompt | llm
    
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
    
    # Crear memoria con ventana de 2 turnos (4 mensajes)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=2,  # Número de turnos a recordar (2 turnos = 4 mensajes)
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    chain = prompt | llm
    
    # Simular conversación larga
    print("\n--- Conversación Larga (k=2) ---")
    
    conversaciones = [
        "Hola, ¿qué tal?",
        "Bien, gracias. ¿Te gusta el fútbol?",
        "Sí, me encanta. ¿Y a ti?",
        "También me gusta. ¿Qué equipo prefieres?",
        "Prefiero el Real Madrid. ¿Y tú?",
        "Yo prefiero el Barcelona.",
        "¿Quién ganó el último clásico?",  # ¿Recordará el contexto?
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
        
        # Cargar historial
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # Formatear prompt
        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=input_usuario
        )
        
        # Obtener respuesta
        response = self.chain.invoke(messages)
        
        # Guardar en memoria
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
        k=3  # Recordar últimos 3 turnos
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
    
    # Turno 5 (ya debería olvidar el turno 1)
    print("\n--- Turno 5 (k=3, ya olvidó turno 1) ---")
    respuesta = asistente.preguntar("¿De qué hablábamos al principio?")
    print(f"R: {respuesta[:100]}...")
    
    # Limpiar memoria
    asistente.limpiar_memoria()


# ============================================================
# 4. MEMORIA CON VERSATILE CHAT HISTORY
# ============================================================

def memoria_con_formato_personalizado():
    """Usar memoria con formato personalizado"""
    
    print("\n" + "=" * 60)
    print("MEMORIA CON FORMATO PERSONALIZADO")
    print("=" * 60)
    
    from langchain.memory import ConversationBufferMemory
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    
    # Agregar mensajes manualmente
    memory.save_context(
        {"input": "Hola, soy María"},
        {"output": "¡Hola María! ¿En qué puedo ayudarte?"}
    )
    
    memory.save_context(
        {"input": "Necesito ayuda con un contrato"},
        {"output": "Claro, ¿qué tipo de contrato necesitas?"}
    )
    
    # Ver historial
    historial = memory.load_memory_variables({})["chat_history"]
    
    print("Historial en formato mensajes:")
    for msg in historial:
        print(f"  {type(msg).__name__}: {msg.content}")
    
    # También se puede obtener como string
    memory_str = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,  # Retorna como string
    )
    
    memory_str.save_context(
        {"input": "Hola"},
        {"output": "Hola, ¿cómo estás?"}
    )
    
    print("\nHistorial como string:")
    print(memory_str.load_memory_variables({})["chat_history"])


if __name__ == "__main__":
    conversation_buffer_memory()
    conversation_window_memory()
    usar_asistente_con_memoria()
    memoria_con_formato_personalizado()

```

---

## 2.1.3 Tipos de Memoria a Corto Plazo

### 1. ConversationBufferMemory

Guarda **TODO** el historial:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)
```

**Pros**: 
- Contexto completo
- El modelo recuerda todo

**Contras**: 
- Consume muchos tokens
- Lento en conversaciones largas

### 2. ConversationBufferWindowMemory

Guarda solo los **últimos N turnos**:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,  # Últimos 5 turnos (10 mensajes)
)
```

**Pros**: 
- Control de tokens
- Más rápido

**Contras**: 
- Pierde contexto antiguo

---

## 2.1.4 Mejores Prácticas

### ✅ DO

```python
# 1. Usa WindowMemory para producción
memory = ConversationBufferWindowMemory(k=5)

# 2. Especifica siempre return_messages=True
memory = ConversationBufferMemory(return_messages=True)

# 3. Limpia la memoria cuando sea necesario
memory.clear()

# 4. Usa MessagesPlaceholder en el prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres útil"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
```

### ❌ DON'T

```python
# 1. No uses BufferMemory sin límite en producción
memory = ConversationBufferMemory()  # ❌ Crece infinitamente

# 2. No olvides guardar el contexto
memory.save_context({"input": x}, {"output": y})  # ❌ Si no guardas, no recuerda

# 3. No uses memory_key incorrecto
memory = ConversationBufferMemory(memory_key="wrong_name")
# prompt debe usar el mismo nombre
```

---

## 2.1.5 Ejercicios Prácticos

### Ejercicio 1: Asistente Personal

Crea un asistente que:
- Recorde tu nombre
- Recorde tus preferencias (comida, color, hobby)
- Responda preguntas basándose en lo recordado

### Ejercicio 2: Window Memory Experiment

Prueba con diferentes valores de k:
- k=1, k=3, k=5, k=10
- Haz 10 preguntas
- Observa cuándo se olvida el contexto

### Ejercicio 3: Multi-Sesión

Crea una clase que:
- Guarde el historial en un archivo JSON
- Permita cargar sesiones anteriores
- Persista la memoria entre ejecuciones

---

## 2.1.6 Recursos Adicionales

### Documentación Oficial
- [LangChain Memory](https://docs.langchain.com/oss/python/langchain/concepts/memory)
- [ConversationBufferMemory](https://docs.langchain.com/oss/python/langchain/integrations/memory/buffer)
- [ConversationBufferWindowMemory](https://docs.langchain.com/oss/python/langchain/integrations/memory/buffer_window)

### Siguiente Lección
➡️ **2.2 Memoria a Largo Plazo**

---

*Lección creada: 2026-03-29*
