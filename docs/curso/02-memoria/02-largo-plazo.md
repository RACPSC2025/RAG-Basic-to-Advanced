# Módulo 2.2: Memoria a Largo Plazo

## Objetivos
- Comprender la memoria a largo plazo
- Usar ConversationSummaryMemory
- Implementar persistencia de conversaciones
- Guardar/cargar memoria desde archivos

---

## 2.2.1 ¿Qué es Memoria a Largo Plazo?

La **memoria a largo plazo** permite recordar información más allá de la sesión actual.

| Tipo | Duración | Uso | Ejemplo |
|------|----------|-----|---------|
| **Corto Plazo** | Sesión actual | Contexto inmediato | "¿Cuál es mi nombre?" (dicho hace 5 turnos) |
| **Largo Plazo** | Múltiples sesiones | Información persistente | "Soy abogado" (dicho hace 3 días) |

---

## 2.2.2 ConversationSummaryMemory

En lugar de guardar todo el historial, guarda un **resumen**:

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
)
```

**Ventajas**:
- Consume menos tokens
- Mantiene esencia de conversaciones largas
- Ideal para largo plazo

---

## 2.2.3 Código de Ejemplo

Archivo: `src/course_examples/modulo_02/02_memoria_largo_plazo.py`

```python
"""
02_memoria_largo_plazo.py
Memoria a Largo Plazo con LangChain

Objetivo: Dominar ConversationSummaryMemory y persistencia
"""

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import (
    ConversationSummaryMemory,
    ConversationBufferMemory,
)
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
)


# ============================================================
# 1. CONVERSATION SUMMARY MEMORY
# ============================================================

def conversation_summary_memory():
    """Memoria que resume la conversación en lugar de guardar todo"""
    
    print("=" * 60)
    print("CONVERSATION SUMMARY MEMORY")
    print("=" * 60)
    
    # Crear memoria con resumen
    memory = ConversationSummaryMemory(
        llm=llm,  # Necesita LLM para generar resúmenes
        memory_key="chat_history",
        return_messages=True,
    )
    
    # Crear prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente legal."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    chain = prompt | llm
    
    # Simular conversación
    print("\n--- Conversación ---")
    
    conversaciones = [
        "Hola, soy Juan Pérez, abogado de profesión.",
        "Trabajo en un bufete en Bogotá especializado en derecho laboral.",
        "Tengo un cliente que fue despedido injustificadamente.",
        "¿Qué acciones legales recomendarías?",
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
    
    # Ver resumen generado
    print("--- Resumen Generado ---")
    resumen = memory.load_memory_variables({})
    print(f"Resumen: {resumen['chat_history'][:500]}...")
    
    # Nueva pregunta después del resumen
    print("\n--- Nueva Pregunta (con contexto resumido) ---")
    input_usuario = "¿Recuerdas cómo me llamo?"
    
    messages = prompt.format_messages(
        chat_history=memory.load_memory_variables({})["chat_history"],
        input=input_usuario
    )
    response = llm.invoke(messages)
    
    print(f"U: {input_usuario}")
    print(f"AI: {response.content}")


# ============================================================
# 2. PERSISTENCIA DE MEMORIA (ARCHIVO JSON)
# ============================================================

class MemoriaPersistente:
    """Memoria que se guarda en archivo JSON"""
    
    def __init__(self, archivo_path: str, system_prompt: str = "Eres un asistente útil."):
        self.archivo_path = archivo_path
        self.system_prompt = system_prompt
        
        # Memoria buffer
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        
        # Cargar memoria existente
        self.cargar_memoria()
        
        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        self.chain = self.prompt | llm
    
    def preguntar(self, input_usuario: str) -> str:
        """Hacer pregunta y guardar en memoria persistente"""
        
        messages = self.prompt.format_messages(
            chat_history=self.memory.load_memory_variables({})["chat_history"],
            input=input_usuario
        )
        
        response = llm.invoke(messages)
        
        # Guardar en memoria
        self.memory.save_context(
            {"input": input_usuario},
            {"output": response.content}
        )
        
        # Persistir a archivo
        self.guardar_memoria()
        
        return response.content
    
    def guardar_memoria(self):
        """Guardar memoria en archivo JSON"""
        
        historial = self.memory.load_memory_variables({})["chat_history"]
        
        # Convertir mensajes a formato serializable
        historial_serializable = []
        for msg in historial:
            historial_serializable.append({
                "tipo": type(msg).__name__,
                "contenido": msg.content
            })
        
        # Guardar en JSON
        with open(self.archivo_path, 'w', encoding='utf-8') as f:
            json.dump({
                "system_prompt": self.system_prompt,
                "historial": historial_serializable
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Memoria guardada en {self.archivo_path}")
    
    def cargar_memoria(self):
        """Cargar memoria desde archivo JSON"""
        
        if not os.path.exists(self.archivo_path):
            print("📭 No hay memoria previa")
            return
        
        try:
            with open(self.archivo_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            
            # Reconstruir historial
            historial = []
            for msg_data in datos["historial"]:
                if msg_data["tipo"] == "HumanMessage":
                    historial.append(HumanMessage(content=msg_data["contenido"]))
                elif msg_data["tipo"] == "AIMessage":
                    historial.append(AIMessage(content=msg_data["contenido"]))
            
            # Cargar en memoria
            self.memory.save_context(
                {"input": "[historial cargado]"},
                {"output": "[historial cargado]"}
            )
            # Limpiar el dummy y cargar real
            self.memory.clear()
            for i in range(0, len(historial), 2):
                if i + 1 < len(historial):
                    self.memory.save_context(
                        {"input": historial[i].content},
                        {"output": historial[i+1].content}
                    )
            
            print(f"✅ Memoria cargada desde {self.archivo_path}")
            print(f"   {len(historial)} mensajes recuperados")
            
        except Exception as e:
            print(f"❌ Error cargando memoria: {e}")
    
    def limpiar_memoria(self):
        """Limpiar memoria y archivo"""
        
        self.memory.clear()
        
        if os.path.exists(self.archivo_path):
            os.remove(self.archivo_path)
            print(f"🗑️ Archivo {self.archivo_path} eliminado")
    
    def ver_historial(self):
        """Ver historial actual"""
        
        historial = self.memory.load_memory_variables({})["chat_history"]
        
        if not historial:
            print("📭 Sin historial")
            return
        
        print(f"📜 Historial ({len(historial)} mensajes):")
        for msg in historial:
            emoji = "👤" if isinstance(msg, HumanMessage) else "🤖"
            print(f"  {emoji} {msg.content[:60]}...")


def usar_memoria_persistente():
    """Demostrar memoria persistente"""
    
    print("=" * 60)
    print("MEMORIA PERSISTENTE (JSON)")
    print("=" * 60)
    
    archivo = "memoria_test.json"
    
    # Primera sesión
    print("\n--- SESIÓN 1 ---")
    asistente1 = MemoriaPersistente(
        archivo_path=archivo,
        system_prompt="Eres un asistente legal colombiano."
    )
    
    asistente1.preguntar("Hola, me llamo María González")
    asistente1.preguntar("Soy abogada especializada en derecho de familia")
    asistente1.preguntar("¿Qué es una demanda de divorcio?")
    
    asistente1.ver_historial()
    
    # Segunda sesión (carga memoria previa)
    print("\n--- SESIÓN 2 (carga memoria anterior) ---")
    asistente2 = MemoriaPersistente(
        archivo_path=archivo,
        system_prompt="Eres un asistente legal colombiano."
    )
    
    asistente2.preguntar("¿Recuerdas mi nombre?")
    asistente2.preguntar("¿En qué soy especialista?")
    
    # Limpiar
    print("\n--- LIMPIEZA ---")
    asistente2.limpiar_memoria()


# ============================================================
# 3. COMBINAR SUMMARY + PERSISTENCIA
# ============================================================

class ResumenPersistente:
    """Combina resumen con persistencia"""
    
    def __init__(self, archivo_path: str):
        self.archivo_path = archivo_path
        
        # Memoria de resumen
        self.memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="resumen",
            return_messages=False,  # Retorna string
        )
        
        # Cargar resumen existente
        self.cargar_resumen()
    
    def agregar_conversacion(self, input_usuario: str, output_ai: str):
        """Agregar conversación al resumen"""
        
        self.memory.save_context(
            {"input": input_usuario},
            {"output": output_ai}
        )
        
        self.guardar_resumen()
    
    def obtener_resumen(self) -> str:
        """Obtener resumen actual"""
        return self.memory.load_memory_variables({})["resumen"]
    
    def guardar_resumen(self):
        """Guardar resumen en archivo"""
        
        resumen = self.obtener_resumen()
        
        with open(self.archivo_path, 'w', encoding='utf-8') as f:
            json.dump({"resumen": resumen}, f, ensure_ascii=False, indent=2)
    
    def cargar_resumen(self):
        """Cargar resumen desde archivo"""
        
        if not os.path.exists(self.archivo_path):
            print("📭 Sin resumen previo")
            return
        
        with open(self.archivo_path, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        # Reconstruir resumen
        resumen = datos.get("resumen", "")
        
        # La memoria de resumen necesita contexto especial
        # Simplificación: solo mostramos el resumen cargado
        print(f"✅ Resumen cargado: {resumen[:100]}...")
    
    def limpiar(self):
        """Limpiar resumen"""
        self.memory.clear()
        if os.path.exists(self.archivo_path):
            os.remove(self.archivo_path)


# ============================================================
# 4. COMPARACIÓN: BUFFER vs SUMMARY
# ============================================================

def comparar_buffer_vs_summary():
    """Comparar Buffer Memory vs Summary Memory"""
    
    print("=" * 60)
    print("COMPARACIÓN: BUFFER vs SUMMARY")
    print("=" * 60)
    
    # Conversación larga para probar
    conversaciones = [
        ("Hola, soy Pedro", "¡Hola Pedro! ¿En qué puedo ayudarte?"),
        ("Tengo 35 años", "Entiendo, tienes 35 años."),
        ("Vivo en Medellín", "¡Medellín es hermosa!"),
        ("Trabajo como ingeniero", "La ingeniería es una gran profesión."),
        ("Tengo dos hijos", "¡Qué bonito! Ser padre es maravilloso."),
        ("Me gusta el fútbol", "¿Qué equipo te gusta?"),
        ("Soy del Nacional", "¡El Verde es grande!"),
    ]
    
    # Buffer Memory
    print("\n--- BUFFER MEMORY ---")
    buffer = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    
    for input_u, output_ai in conversaciones:
        buffer.save_context({"input": input_u}, {"output": output_ai})
    
    buffer_historial = buffer.load_memory_variables({})["chat_history"]
    print(f"Mensajes: {len(buffer_historial)}")
    print(f"Tokens aproximados: {sum(len(msg.content) for msg in buffer_historial) // 4}")
    
    # Summary Memory
    print("\n--- SUMMARY MEMORY ---")
    summary = ConversationSummaryMemory(
        llm=llm,
        memory_key="resumen",
        return_messages=False,
    )
    
    for input_u, output_ai in conversaciones:
        summary.save_context({"input": input_u}, {"output": output_ai})
    
    resumen = summary.load_memory_variables({})["resumen"]
    print(f"Resumen: {resumen[:300]}...")
    print(f"Tokens aproximados: {len(resumen) // 4}")


if __name__ == "__main__":
    conversation_summary_memory()
    usar_memoria_persistente()
    comparar_buffer_vs_summary()

```

---

## 2.2.4 Cuándo Usar Cada Tipo

| Escenario | Tipo Recomendado |
|-----------|------------------|
| Chatbot simple | BufferMemory |
| Conversaciones largas | WindowMemory (k=5-10) |
| Asistente personal | SummaryMemory |
| Multi-sesión | Persistencia JSON |
| Producción con límites | WindowMemory + Summary |

---

## 2.2.5 Ejercicios Prácticos

### Ejercicio 1: Diario Personal

Crea un asistente que:
- Guarde eventos importantes de tu día
- Resuma la semana
- Permita preguntar "¿Qué hice el lunes?"

### Ejercicio 2: Multi-Usuario

Crea un sistema que:
- Guarde memoria separada por usuario
- Use ID de usuario como clave
- Permita cambiar entre usuarios

### Ejercicio 3: Resumen Automático

Implementa:
- Resumen cada 10 turnos
- Guarda solo el resumen
- Permite recuperar contexto

---

## 2.2.6 Recursos Adicionales

### Documentación Oficial
- [ConversationSummaryMemory](https://docs.langchain.com/oss/python/langchain/integrations/memory/summary)
- [Memory Persistence](https://docs.langchain.com/oss/python/langchain/concepts/memory#persistence)
- [Custom Memory](https://docs.langchain.com/oss/python/langchain/how_to/custom_memory)

### Siguiente Módulo
➡️ **Módulo 3: Streaming**

---

*Lección creada: 2026-03-29*
