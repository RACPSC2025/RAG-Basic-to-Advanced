# Módulo 5.1: Herramientas (Tools) - Guía Completa

## Objetivos
- Comprender qué son las herramientas y por qué son importantes
- Crear herramientas en LangChain, LangGraph y LlamaIndex
- Entender las diferencias entre los tres frameworks
- Implementar herramientas con estado, memoria y streaming
- Casos de uso reales para RAG legal

---

## 5.1.1 ¿Qué son las Herramientas?

Las **herramientas** permiten que los agentes interactúen con sistemas externos ejecutando funciones que tú defines.

```
┌─────────────────────────────────────────────────────────┐
│                 AGENTE SIN HERRAMIENTAS                 │
├─────────────────────────────────────────────────────────┤
│  Usuario: "¿Cuál es el clima en Bogotá?"                │
│  LLM: "No puedo acceder a información en tiempo real."  │
│                                                         │
│  ❌ Limitado al conocimiento de entrenamiento           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 AGENTE CON HERRAMIENTAS                 │
├─────────────────────────────────────────────────────────┤
│  Usuario: "¿Cuál es el clima en Bogotá?"                │
│  LLM: [Llama herramienta: get_weather(Bogotá)]          │
│  Herramienta: "22°C, soleado"                           │
│  LLM: "En Bogotá están 22°C y soleado ☀️"               │
│                                                         │
│  ✅ Acceso a datos en tiempo real, APIs, bases de datos │
└─────────────────────────────────────────────────────────┘
```

---

## 5.1.2 Comparativa: LangChain vs LangGraph vs LlamaIndex

| Característica | LangChain | LangGraph | LlamaIndex |
|----------------|-----------|-----------|------------|
| **Creación** | `@tool` decorator | `@tool` + `ToolNode` | `FunctionTool.from_defaults()` |
| **Schema** | Type hints + docstring | Igual que LangChain | Type hints + docstring |
| **Estado** | `ToolRuntime` | State injection | Menos integrado |
| **Streaming** | `stream_writer` | Vía ToolNode | Limitado |
| **Memoria** | `BaseStore` integrada | Vía State/Store | `return_direct` |
| **Error handling** | Configurable | ToolNode maneja errores | Básico |
| **Use case ideal** | Agentes simples | Agentes complejos | RAG + queries |

---

## 5.1.3 Código de Ejemplo Comparativo

Archivo: `src/course_examples/modulo_05/01_tools_comparativa.py`

```python
"""
01_tools_comparativa.py
Comparativa de Herramientas: LangChain vs LangGraph vs LlamaIndex

Objetivo: Entender las diferencias y similitudes entre frameworks
"""

import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

# ==========================================
# CONFIGURACIÓN COMÚN
# ==========================================

load_dotenv()

# Base de datos simulada de leyes colombianas
BASE_DATOS_LEYES = {
    "tutela": {
        "nombre": "Acción de Tutela",
        "descripcion": "Mecanismo para proteger derechos fundamentales",
        "articulo": "Artículo 86 Constitución",
        "tiempo_respuesta": "10 días hábiles"
    },
    "derecho_peticion": {
        "nombre": "Derecho de Petición",
        "descripcion": "Derecho fundamental para solicitar información",
        "articulo": "Artículo 23 Constitución",
        "tiempo_respuesta": "15 días hábiles"
    },
    "habeas_corpus": {
        "nombre": "Habeas Corpus",
        "descripcion": "Protección a la libertad personal",
        "articulo": "Artículo 30 Constitución",
        "tiempo_respuesta": "36 horas"
    }
}


# ==========================================
# 1. LANGCHAIN TOOLS
# ==========================================

def langchain_tools_demo():
    """Demostrar herramientas en LangChain puro"""
    
    print("=" * 80)
    print("1. LANGCHAIN TOOLS")
    print("=" * 80)
    
    from langchain.tools import tool
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Herramienta básica con @tool
    @tool
    def buscar_ley(nombre: str) -> str:
        """Busca información sobre una ley o mecanismo legal colombiano.
        
        Args:
            nombre: Nombre de la ley o mecanismo (ej: 'tutela', 'derecho_peticion')
        
        Returns:
            Información completa de la ley
        """
        nombre_key = nombre.lower().replace(" ", "_")
        
        if nombre_key in BASE_DATOS_LEYES:
            ley = BASE_DATOS_LEYES[nombre_key]
            return f"""
            Ley: {ley['nombre']}
            Descripción: {ley['descripcion']}
            Base Legal: {ley['articulo']}
            Tiempo de respuesta: {ley['tiempo_respuesta']}
            """
        else:
            return f"Ley '{nombre}' no encontrada. Opciones: tutela, derecho_peticion, habeas_corpus"
    
    # Herramienta con schema personalizado (Pydantic)
    class CompararLeyesInput(BaseModel):
        ley1: str = Field(description="Primera ley a comparar")
        ley2: str = Field(description="Segunda ley a comparar")
    
    @tool(args_schema=CompararLeyesInput)
    def comparar_leyes(ley1: str, ley2: str) -> str:
        """Compara dos leyes colombianas y muestra sus diferencias."""
        
        key1 = ley1.lower().replace(" ", "_")
        key2 = ley2.lower().replace(" ", "_")
        
        if key1 not in BASE_DATOS_LEYES or key2 not in BASE_DATOS_LEYES:
            return "Una o ambas leyes no existen en la base de datos"
        
        ley1_data = BASE_DATOS_LEYES[key1]
        ley2_data = BASE_DATOS_LEYES[key2]
        
        return f"""
        COMPARACIÓN:
        
        {ley1_data['nombre']} vs {ley2_data['nombre']}
        
        Tiempos de respuesta:
        - {ley1_data['nombre']}: {ley1_data['tiempo_respuesta']}
        - {ley2_data['nombre']}: {ley2_data['tiempo_respuesta']}
        
        Bases legales:
        - {ley1_data['nombre']}: {ley1_data['articulo']}
        - {ley2_data['nombre']}: {ley2_data['articulo']}
        """
    
    # Crear agente con herramientas
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    )
    
    agent = create_agent(
        llm,
        tools=[buscar_ley, comparar_leyes],
        system_prompt="Eres un asistente legal experto en derecho colombiano. Usa las herramientas para obtener información precisa."
    )
    
    # Probar el agente
    print("\n--- Prueba 1: Búsqueda simple ---")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "¿Qué es una tutela y cuánto tiempo tienen para responder?"}]
    })
    print(f"Respuesta: {response['messages'][-1].content[:300]}...")
    
    print("\n--- Prueba 2: Comparación ---")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Compara la tutela con el derecho de petición"}]
    })
    print(f"Respuesta: {response['messages'][-1].content[:300]}...")
    
    # Mostrar schema de la herramienta
    print("\n--- Schema de buscar_ley ---")
    print(f"Nombre: {buscar_ley.name}")
    print(f"Descripción: {buscar_ley.description}")
    print(f"Args schema: {buscar_ley.args}")


# ==========================================
# 2. LANGGRAPH TOOLS + TOOLNODE
# ==========================================

def langgraph_tools_demo():
    """Demostrar herramientas en LangGraph con ToolNode"""
    
    print("\n" + "=" * 80)
    print("2. LANGGRAPH TOOLS + TOOLNODE")
    print("=" * 80)
    
    from langchain.tools import tool
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Mismas herramientas que LangChain
    @tool
    def buscar_ley(nombre: str) -> str:
        """Busca información sobre una ley o mecanismo legal colombiano."""
        nombre_key = nombre.lower().replace(" ", "_")
        
        if nombre_key in BASE_DATOS_LEYES:
            ley = BASE_DATOS_LEYES[nombre_key]
            return f"{ley['nombre']}: {ley['descripcion']} ({ley['tiempo_respuesta']})"
        return "Ley no encontrada"
    
    @tool
    def calcular_tiempo_procesal(dias: int, tipo: Literal["habiles", "calendario"]) -> str:
        """Calcula fechas en derecho procesal colombiano.
        
        Args:
            dias: Número de días
            tipo: Tipo de días (hábiles o calendario)
        """
        # Simplificación: solo suma días
        from datetime import datetime, timedelta
        hoy = datetime.now()
        resultado = hoy + timedelta(days=dias)
        
        return f"Hoy: {hoy.strftime('%Y-%m-%d')} → {dias} días {tipo}: {resultado.strftime('%Y-%m-%d')}"
    
    # ToolNode: maneja ejecución de herramientas
    tool_node = ToolNode([buscar_ley, calcular_tiempo_procesal])
    
    # Crear grafo personalizado
    builder = StateGraph(MessagesState)
    
    # Nodo de agente (LLM decide si usar herramientas)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    ).bind_tools([buscar_ley, calcular_tiempo_procesal])
    
    def agente(state: MessagesState):
        """Nodo que decide si llamar herramientas"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    # Condición: ¿el LLM llamó herramientas?
    def debe_ejecutar_herramientas(state: MessagesState) -> str:
        ultimo_mensaje = state["messages"][-1]
        
        # Si tiene tool_calls, ejecutar herramientas
        if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
            return "herramientas"
        return "fin"
    
    # Construir grafo
    builder.add_node("agente", agente)
    builder.add_node("herramientas", tool_node)
    
    builder.add_edge(START, "agente")
    builder.add_conditional_edges(
        "agente",
        debe_ejecutar_herramientas,
        {
            "herramientas": "herramientas",
            "fin": END
        }
    )
    
    # Después de herramientas, volver al agente
    builder.add_edge("herramientas", "agente")
    
    graph = builder.compile()
    
    # Probar
    print("\n--- Prueba 1: Búsqueda ---")
    response = graph.invoke({
        "messages": [HumanMessage(content="¿Qué es un habeas corpus?")]
    })
    print(f"Respuesta: {response['messages'][-1].content[:200]}...")
    
    print("\n--- Prueba 2: Cálculo ---")
    response = graph.invoke({
        "messages": [HumanMessage(content="Si tengo 10 días hábiles a partir de hoy, qué fecha sería?")]
    })
    print(f"Respuesta: {response['messages'][-1].content[:200]}...")
    
    print("\n✅ LangGraph permite control total del flujo con ToolNode")


# ==========================================
# 3. LLAMAINDEX TOOLS
# ==========================================

def llamaindex_tools_demo():
    """Demostrar herramientas en LlamaIndex"""
    
    print("\n" + "=" * 80)
    print("3. LLAMAINDEX TOOLS")
    print("=" * 80)
    
    try:
        from llama_index.core.tools import FunctionTool
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.google_genai import GoogleGenAI
        
        # Herramienta básica
        def buscar_ley(nombre: str) -> str:
            """Busca información sobre una ley o mecanismo legal colombiano.
            
            Args:
                nombre: Nombre de la ley (tutela, derecho_peticion, habeas_corpus)
            """
            nombre_key = nombre.lower().replace(" ", "_")
            
            if nombre_key in BASE_DATOS_LEYES:
                ley = BASE_DATOS_LEYES[nombre_key]
                return f"{ley['nombre']}: {ley['descripcion']}"
            return "Ley no encontrada. Opciones: tutela, derecho_peticion, habeas_corpus"
        
        # Herramienta con anotaciones
        from typing import Annotated
        
        def comparar_leyes(
            ley1: Annotated[str, "Primera ley a comparar"],
            ley2: Annotated[str, "Segunda ley a comparar"]
        ) -> str:
            """Compara dos leyes colombianas."""
            
            key1 = ley1.lower().replace(" ", "_")
            key2 = ley2.lower().replace(" ", "_")
            
            if key1 not in BASE_DATOS_LEYES or key2 not in BASE_DATOS_LEYES:
                return "Una o ambas leyes no existen"
            
            return f"Comparación: {BASE_DATOS_LEYES[key1]['nombre']} vs {BASE_DATOS_LEYES[key2]['nombre']}"
        
        # Convertir a FunctionTool
        tool_buscar = FunctionTool.from_defaults(buscar_ley)
        tool_comparar = FunctionTool.from_defaults(comparar_leyes)
        
        # Crear agente
        llm = GoogleGenAI(model="gemini-2.0-flash-exp")
        
        agent = FunctionAgent(
            llm=llm,
            tools=[tool_buscar, tool_comparar]
        )
        
        # Probar
        print("\n--- Prueba 1: Búsqueda ---")
        response = agent.run("¿Qué es una tutela?")
        print(f"Respuesta: {str(response)[:200]}...")
        
        # return_direct=True: termina el agente inmediatamente
        def respuesta_directa(mensaje: str) -> str:
            """Responde directamente sin más procesamiento del agente."""
            return f"RESPUESTA DIRECTA: {mensaje}"
        
        tool_directo = FunctionTool.from_defaults(
            respuesta_directa,
            return_direct=True  # ⚡ Termina el agente
        )
        
        agent_directo = FunctionAgent(
            llm=llm,
            tools=[tool_buscar, tool_directo]
        )
        
        print("\n--- Prueba 2: Return Direct ---")
        response = agent_directo.run("Usa respuesta directa: Hola mundo")
        print(f"Respuesta: {str(response)[:200]}...")
        
        # Debug: ver schema
        print("\n--- Schema de herramienta ---")
        schema = tool_buscar.metadata.get_parameters_dict()
        print(f"Schema: {schema}")
        
    except ImportError as e:
        print(f"⚠️ LlamaIndex no instalado: {e}")
        print("   Instalar: pip install llama-index llama-index-llms-google-genai")


# ==========================================
# 4. HERRAMIENTAS CON ESTADO (LANGCHAIN)
# ==========================================

def herramientas_con_estado():
    """Herramientas que acceden y modifican estado"""
    
    print("\n" + "=" * 80)
    print("4. HERRAMIENTAS CON ESTADO")
    print("=" * 80)
    
    from langchain.tools import tool, ToolRuntime
    from langgraph.types import Command
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Herramienta que lee estado
    @tool
    def obtener_preferencias_usuario(runtime: ToolRuntime) -> str:
        """Obtiene las preferencias del usuario del estado."""
        
        # Acceder a estado personalizado
        preferencias = runtime.state.get("user_preferences", {})
        
        if not preferencias:
            return "No hay preferencias guardadas"
        
        return f"Preferencias: {', '.join(f'{k}={v}' for k, v in preferencias.items())}"
    
    # Herramienta que modifica estado
    @tool
    def guardar_preferencia(clave: str, valor: str) -> Command:
        """Guarda una preferencia del usuario en el estado."""
        
        return Command(
            update={
                "user_preferences": {clave: valor}
            }
        )
    
    # Herramienta que usa contexto (user_id)
    from dataclasses import dataclass
    
    @dataclass
    class UserContext:
        user_id: str
        rol: str
    
    @tool
    def obtener_info_usuario(runtime: ToolRuntime) -> str:
        """Obtiene información del usuario desde el contexto."""
        
        user_id = runtime.context.user_id
        rol = runtime.context.rol
        
        return f"Usuario: {user_id}, Rol: {rol}"
    
    # Crear agente con contexto
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    )
    
    agent = create_agent(
        llm,
        tools=[obtener_preferencias_usuario, guardar_preferencia, obtener_info_usuario],
        context_schema=UserContext,
    )
    
    # Probar con contexto
    print("\n--- Prueba con contexto ---")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "¿Cuál es mi información de usuario?"}]},
        context=UserContext(user_id="usuario_123", rol="abogado")
    )
    print(f"Respuesta: {response['messages'][-1].content[:200]}...")


# ==========================================
# 5. HERRAMIENTAS CON STREAMING
# ==========================================

def herramientas_con_streaming():
    """Herramientas que emiten updates en tiempo real"""
    
    print("\n" + "=" * 80)
    print("5. HERRAMIENTAS CON STREAMING")
    print("=" * 80)
    
    from langchain.tools import tool, ToolRuntime
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    import time
    
    @tool
    def proceso_largo(cantidad: int, runtime: ToolRuntime) -> str:
        """Simula un proceso largo con updates en tiempo real."""
        
        writer = runtime.stream_writer
        
        writer(f"🚀 Iniciando proceso de {cantidad} pasos...")
        
        for i in range(cantidad):
            time.sleep(0.5)  # Simular trabajo
            writer(f"⚙️ Paso {i+1}/{cantidad} completado")
        
        writer("✅ Proceso completado")
        
        return f"Proceso completado: {cantidad} pasos ejecutados"
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    )
    
    agent = create_agent(
        llm,
        tools=[proceso_largo],
    )
    
    print("\n--- Prueba con streaming ---")
    
    # Streaming
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "Ejecuta un proceso de 5 pasos"}]
    }):
        # Verificar si hay stream events
        if "messages" in chunk:
            for msg in chunk["messages"]:
                if hasattr(msg, "content") and msg.content:
                    print(f"  {msg.content[:100]}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("🎓 CURSO LANGCHAIN/LANGGRAPH - MÓDULO 5.1: HERRAMIENTAS\n")
    
    # 1. LangChain
    langchain_tools_demo()
    
    # 2. LangGraph
    langgraph_tools_demo()
    
    # 3. LlamaIndex
    llamaindex_tools_demo()
    
    # 4. Estado
    herramientas_con_estado()
    
    # 5. Streaming
    herramientas_con_streaming()
    
    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO")
    print("=" * 80)
    print("""
    ┌─────────────────────┬──────────────┬──────────────┬──────────────┐
    │ Característica      │ LangChain    │ LangGraph    │ LlamaIndex   │
    ├─────────────────────┼──────────────┼──────────────┼──────────────┤
    │ Facilidad de uso    │ ⭐⭐⭐⭐⭐      │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │
    │ Control de flujo    │ ⭐⭐⭐         │ ⭐⭐⭐⭐⭐      │ ⭐⭐⭐         │
    │ Estado integrado    │ ⭐⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐      │ ⭐⭐          │
    │ Streaming           │ ⭐⭐⭐⭐⭐      │ ⭐⭐⭐⭐       │ ⭐⭐          │
    │ RAG integration     │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐⭐      │
    │ Error handling      │ ⭐⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐      │ ⭐⭐⭐         │
    └─────────────────────┴──────────────┴──────────────┴──────────────┘
    
    RECOMENDACIÓN:
    - LangChain: Agentes simples con herramientas
    - LangGraph: Agentes complejos con flujo personalizado
    - LlamaIndex: RAG + queries sobre documentos
    """)

```

---

## 5.1.4 Creación de Herramientas: Sintaxis

### LangChain (@tool)

```python
from langchain.tools import tool

@tool
def mi_herramienta(param: str) -> str:
    """Descripción clara para el LLM."""
    return f"Resultado: {param}"
```

### LangGraph (ToolNode)

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def herramienta1(x: int) -> str:
    return str(x)

tool_node = ToolNode([herramienta1])
```

### LlamaIndex (FunctionTool)

```python
from llama_index.core.tools import FunctionTool

def mi_funcion(x: str) -> str:
    """Descripción."""
    return x

tool = FunctionTool.from_defaults(mi_funcion)
```

---

## 5.1.5 Ejercicios Prácticos

### Ejercicio 1: Calculadora Legal

Crea herramientas para:
- Calcular intereses moratorios
- Convertir días hábiles a calendario
- Calcular fechas de vencimiento

### Ejercicio 2: Buscador de Jurisprudencia

Implementa:
- Búsqueda por palabra clave
- Filtro por corte/tribunal
- Filtro por fecha

### Ejercicio 3: Herramienta con Memoria

Crea una herramienta que:
- Guarde consultas frecuentes del usuario
- Recomiende basándose en historial
- Use `BaseStore` para persistencia

---

## 5.1.6 Recursos Adicionales

### Documentación Oficial
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangGraph ToolNode](https://docs.langchain.com/oss/python/langgraph/how_to/tool_node)
- [LlamaIndex Tools](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools/)

### Siguiente Lección
➡️ **5.2 Tool Calling con LLMs**

---

*Lección creada: 2026-03-29*
