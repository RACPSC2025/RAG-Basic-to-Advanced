# Módulo 3.1: Streaming de Tokens

## Objetivos
- Comprender el streaming en LangChain
- Implementar streaming de tokens en tiempo real
- Manejar callbacks para eventos
- Mejorar la experiencia de usuario con streaming

---

## 3.1.1 ¿Qué es Streaming?

**Streaming** permite recibir la respuesta del LLM token por token, en tiempo real, en lugar de esperar toda la respuesta.

```
┌─────────────────────────────────────────────────────────┐
│              SIN STREAMING                              │
├─────────────────────────────────────────────────────────┤
│  Usuario envía pregunta...                              │
│  ⏳ Espera 5 segundos...                                │
│  ✅ Recibe respuesta completa de golpe                  │
│                                                         │
│  Experiencia: Usuario espera sin feedback               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              CON STREAMING                              │
├─────────────────────────────────────────────────────────┤
│  Usuario envía pregunta...                              │
│  ⚡ Token 1: "La"                                       │
│  ⚡ Token 2: " tutela"                                  │
│  ⚡ Token 3: " es"                                      │
│  ⚡ Token 4: " un"                                      │
│  ...                                                    │
│  ✅ Respuesta completa gradualmente                     │
│                                                         │
│  Experiencia: Usuario ve progreso en tiempo real        │
└─────────────────────────────────────────────────────────┘
```

---

## 3.1.2 Código de Ejemplo

Archivo: `src/course_examples/modulo_03/01_streaming.py`

```python
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

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
    streaming=True,  # Habilitar streaming
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
    
    # Usar .stream() en lugar de .invoke()
    start_time = time.time()
    
    for token in chain.stream({}):
        print(token, end="", flush=True)
        # Pequeña pausa para simular efecto typewriter
        time.sleep(0.02)
    
    elapsed = time.time() - start_time
    print(f"\n\n⏱️ Tiempo total: {elapsed:.2f} segundos")


# ============================================================
# 2. STREAMING CON CALLBACKS
# ============================================================

def streaming_con_callbacks():
    """Usar callbacks para eventos de streaming"""
    
    print("\n" + "=" * 60)
    print("STREAMING CON CALLBACKS")
    print("=" * 60)
    
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.callbacks.base import BaseCallbackHandler
    
    # Callback personalizado
    class MiCallback(BaseCallbackHandler):
        def on_llm_start(self, serialized, input_str, **kwargs):
            print("\n🚀 Iniciando generación...")
        
        def on_llm_new_token(self, token: str, **kwargs):
            print(token, end="", flush=True)
        
        def on_llm_end(self, response, **kwargs):
            print("\n✅ Generación completada")
        
        def on_llm_error(self, error, **kwargs):
            print(f"\n❌ Error: {error}")
    
    # Crear LLM con callbacks
    llm_callbacks = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.5,
        streaming=True,
        callbacks=[MiCallback()],
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "¿Cuáles son los elementos del contrato de trabajo?")
    ])
    
    chain = prompt | llm_callbacks | StrOutputParser()
    
    print("\nRespuesta:\n")
    chain.invoke({})


# ============================================================
# 3. STREAMING CON ACUMULADOR
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
    
    # Acumulador
    respuesta_completa = ""
    
    print("\nGenerando respuesta:\n")
    
    for token in chain.stream({}):
        print(token, end="", flush=True)
        respuesta_completa += token
        time.sleep(0.02)
    
    print(f"\n\n✅ Respuesta acumulada ({len(respuesta_completa)} caracteres)")
    print(f"📄 Primeras 100 letras: {respuesta_completa[:100]}...")


# ============================================================
# 4. STREAMING EN CHAT CON HISTORIAL
# ============================================================

def streaming_con_historial():
    """Streaming en conversación con memoria"""
    
    print("\n" + "=" * 60)
    print("STREAMING CON HISTORIAL")
    print("=" * 60)
    
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.prompts import MessagesPlaceholder
    
    # Memoria
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
    )
    
    # Prompt con historial
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente legal colombiano."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Chain
    chain = prompt | llm | StrOutputParser()
    
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
    
    # Turno 2 (con contexto)
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
# 5. STREAMING ASÍNCRONO
# ============================================================

async def streaming_asincrono():
    """Streaming asíncrono para aplicaciones modernas"""
    
    print("\n" + "=" * 60)
    print("STREAMING ASÍNCRONO")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Explica la diferencia entre ley y decreto")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    print("\nRespuesta asíncrona:\n")
    
    # Usar .astream() en lugar de .stream()
    async for token in chain.astream({}):
        print(token, end="", flush=True)
        time.sleep(0.02)


# ============================================================
# 6. CLASE CON STREAMING
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
            
            print()  # Newline
            
        else:
            respuesta = self.chain.invoke({
                "chat_history": self.memory,
                "input": input_usuario
            })
            print(f"\n🤖 AI: {respuesta}")
        
        # Guardar en memoria
        from langchain_core.messages import HumanMessage, AIMessage
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
    
    # Pregunta 1 con streaming
    asistente.preguntar(
        "¿Qué es la Constitución?",
        usar_streaming=True
    )
    
    # Pregunta 2 con streaming (con contexto)
    asistente.preguntar(
        "¿Cuál es la Constitución vigente en Colombia?",
        usar_streaming=True
    )
    
    # Pregunta 3 sin streaming
    asistente.preguntar(
        "¿Cuántos artículos tiene?",
        usar_streaming=False
    )


# ============================================================
# 7. COMPARACIÓN: STREAMING VS NO STREAMING
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
    
    # Comparar
    print(f"\n📊 COMPARACIÓN:")
    print(f"   Sin streaming: {len(respuesta)} chars en {elapsed:.2f}s (espera total)")
    print(f"   Con streaming: {len(respuesta_stream)} chars en {elapsed:.2f}s (progreso visible)")


if __name__ == "__main__":
    streaming_basico()
    streaming_con_callbacks()
    streaming_con_acumulador()
    streaming_con_historial()
    # streaming_asincrono()  # Requiere asyncio
    usar_asistente_streaming()
    comparar_streaming_vs_no_streaming()

```

---

## 3.1.3 Métodos de Streaming

### 1. `.stream()` (Síncrono)

```python
chain = prompt | llm | parser

for token in chain.stream({}):
    print(token, end="", flush=True)
```

### 2. `.astream()` (Asíncrono)

```python
async for token in chain.astream({}):
    print(token, end="", flush=True)
```

### 3. Callbacks

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatGoogleGenerativeAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

---

## 3.1.4 Mejores Prácticas

### ✅ DO

```python
# 1. Habilita streaming en el LLM
llm = ChatGoogleGenerativeAI(streaming=True)

# 2. Usa flush=True para output inmediato
print(token, end="", flush=True)

# 3. Acumula la respuesta si la necesitas después
respuesta = ""
for token in chain.stream({}):
    respuesta += token

# 4. Usa callbacks para lógica compleja
class MiCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        # Lógica custom
        pass
```

### ❌ DON'T

```python
# 1. No uses stream() sin flush
print(token)  # ❌ Bufferizado

# 2. No mezcles invoke() con expectativas de streaming
response = llm.invoke(prompt)  # ❌ No hay streaming

# 3. No olvides habilitar streaming en el LLM
llm = ChatGoogleGenerativeAI(streaming=False)  # ❌ No funcionará
```

---

## 3.1.5 Ejercicios Prácticos

### Ejercicio 1: Typewriter Effect

Crea un efecto de máquina de escribir:
- Cada token aparece con delay aleatorio (0.01-0.05s)
- Sonido beep al final (opcional)

### Ejercicio 2: Progress Bar

Implementa:
- Barra de progreso basada en tokens
- Estimado de tiempo restante
- Contador de tokens por segundo

### Ejercicio 3: Streaming con Cancelación

Crea:
- Botón/código para cancelar streaming
- Guardar lo generado hasta el momento
- Manejar interrupción gracefully

---

## 3.1.6 Recursos Adicionales

### Documentación Oficial
- [LangChain Streaming](https://docs.langchain.com/oss/python/langchain/concepts/streaming)
- [Callbacks](https://docs.langchain.com/oss/python/langchain/concepts/callbacks)
- [Async Streaming](https://docs.langchain.com/oss/python/langchain/how_to/async_streaming)

### Siguiente Lección
➡️ **Módulo 4: Introducción a LangGraph**

---

*Lección creada: 2026-03-29*
