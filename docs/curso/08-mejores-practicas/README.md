# 📖 Guía de Mejores Prácticas - LangChain + LangGraph

> **Recopilación de mejores prácticas de los módulos 1-6**  
> **Actualización**: 2026-03-29

---

## 📋 Índice

1. [Módulo 1: Fundamentos](#módulo-1-fundamentos)
2. [Módulo 2: Memoria](#módulo-2-memoria)
3. [Módulo 3: Streaming](#módulo-3-streaming)
4. [Módulo 4: LangGraph](#módulo-4-langgraph)
5. [Módulo 5: Herramientas](#módulo-5-herramientas)
6. [Módulo 6: Human in the Loop](#módulo-6-human-in-the-loop)
7. [General y Producción](#general-y-producción)

---

## Módulo 1: Fundamentos

### ✅ DO (Haz esto)

```python
# 1. Siempre usa type hints
@tool
def mi_funcion(param: str) -> str:  # ✅

# 2. Docstrings descriptivos
"""
Busca información específica.

Args:
    param: Descripción del parámetro

Returns:
    Información encontrada
"""

# 3. Temperature según el caso
ChatGoogleGenerativeAI(temperature=0.3)  # ✅ Para herramientas/facts
ChatGoogleGenerativeAI(temperature=0.7)  # ✅ Para creativo

# 4. Maneja errores de API
try:
    response = llm.invoke(prompt)
except Exception as e:
    logger.error(f"Error LLM: {e}")
    response = AIMessage(content="Ocurrió un error. Por favor intenta de nuevo.")
```

### ❌ DON'T (No hagas esto)

```python
# 1. Sin type hints
def mi_funcion(param):  # ❌

# 2. Docstring vago
"""Busca algo"""  # ❌

# 3. Temperature alta para facts
ChatGoogleGenerativeAI(temperature=1.5)  # ❌

# 4. Sin manejo de errores
response = llm.invoke(prompt)  # ❌ Sin try/except
```

---

## Módulo 2: Memoria

### ✅ DO

```python
# 1. Usa WindowMemory para conversaciones largas
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5  # Últimos 5 turnos
)

# 2. SummaryMemory para conversaciones muy largas
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="resumen"
)

# 3. Limpia memoria cuando sea necesario
memory.clear()

# 4. Persiste en producción
import json
with open("historial.json", "w") as f:
    json.dump(historial, f)
```

### ❌ DON'T

```python
# 1. BufferMemory sin límite en producción
memory = ConversationBufferMemory()  # ❌ Crece infinitamente

# 2. No guardar contexto
memory.save_context(...)  # ❌ Si no guardas, no recuerda

# 3. Memory_key incorrecto
memory = ConversationBufferMemory(memory_key="wrong")
# prompt debe usar el mismo nombre
```

---

## Módulo 3: Streaming

### ✅ DO

```python
# 1. Habilita streaming en el LLM
llm = ChatGoogleGenerativeAI(streaming=True)

# 2. Usa flush=True para output inmediato
print(token, end="", flush=True)

# 3. Acumula si necesitas el resultado después
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
# 1. Sin flush
print(token)  # ❌ Bufferizado

# 2. Mezclar invoke() con expectativas de streaming
response = llm.invoke(prompt)  # ❌ No hay streaming

# 3. No habilitar streaming
llm = ChatGoogleGenerativeAI(streaming=False)  # ❌
```

---

## Módulo 4: LangGraph

### ✅ DO

```python
# 1. Define estado claramente con TypedDict
class State(TypedDict):
    messages: List[str]
    contador: int

# 2. Usa START y END
builder.add_edge(START, "primer_nodo")
builder.add_edge("ultimo_nodo", END)

# 3. Conditional edges para routing
def router(state: State) -> str:
    if condition:
        return "opcion_a"
    return "opcion_b"

builder.add_conditional_edges("nodo", router, {
    "opcion_a": "nodo_a",
    "opcion_b": "nodo_b"
})

# 4. Compila con checkpointer para HITL
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### ❌ DON'T

```python
# 1. Estado sin tipo
class State:  # ❌ Usa TypedDict
    pass

# 2. Olvidar END
builder.add_edge("nodo", END)  # ❌ Necesario

# 3. Conditional edges sin mapeo completo
builder.add_conditional_edges("nodo", router)  # ❌ Falta mapeo

# 4. Sin checkpointer para HITL
graph = builder.compile()  # ❌ Para HITL necesita checkpointer
```

---

## Módulo 5: Herramientas

### ✅ DO

```python
# 1. Type hints siempre
@tool
def mi_funcion(param: str) -> str:  # ✅

# 2. Field con descripciones
campo: str = Field(description="Descripción clara")

# 3. Names en snake_case
@tool("buscar_ley")  # ✅

# 4. Temperature baja para tools
ChatGoogleGenerativeAI(temperature=0.3)  # ✅

# 5. Docstring informativo para el LLM
"""
Busca información específica sobre X tema.

Args:
    param: Descripción del parámetro

Returns:
    Información encontrada
"""
```

### ❌ DON'T

```python
# 1. Sin type hints
def buscar(nombre):  # ❌

# 2. Docstring vago
"""Busca algo"""  # ❌

# 3. Nombres con espacios
@tool("Buscar Ley")  # ❌

# 4. Temperature alta
ChatGoogleGenerativeAI(temperature=1.5)  # ❌

# 5. Sin validación Pydantic
class MiSchema(BaseModel):
    campo: str  # ❌ Sin Field description
```

---

## Módulo 6: Human in the Loop

### ✅ DO

```python
# 1. Usa checkpointer persistente en producción
checkpointer = PostgresSaver(...)  # ✅
checkpointer = MemorySaver()       # ⚠️ Solo desarrollo

# 2. Proporciona contexto claro en el interrupt
interrupt({
    "pregunta": "¿Qué quieres hacer?",
    "opciones": ["aprobar", "rechazar", "editar"],
    "contexto": state["datos"]
})

# 3. Maneja múltiples interrupts con IDs
resume_map = {id: valor for id, valor in interrupts}

# 4. Valida entrada humana antes de continuar
if not validar_entrada(humano_input):
    return Command(goto="solicitar_nuevamente")

# 5. Usa thread_id consistente
config = {"configurable": {"thread_id": "mi-hilo-123"}}
```

### ❌ DON'T

```python
# 1. Olvidar el checkpointer
graph = builder.compile()  # ❌ Sin checkpointer
# interrupt() no funcionará

# 2. Usar thread_id incorrecto al reanudar
graph.invoke(Command(resume=valor), 
             config={"thread_id": "otro"})  # ❌

# 3. No proporcionar suficiente contexto
interrupt("¿Apruebas?")  # ❌ Muy vago

interrupt({
    "accion": "...",
    "detalles": "...",
    "consecuencias": "..."
})  # ✅

# 4. Envolver interrupt en try/except
try:
    valor = interrupt()  # ❌
except Exception:
    ...

valor = interrupt()  # ✅ Sin try/except
```

---

## General y Producción

### ✅ DO

```python
# 1. Variables de entorno
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. Logging apropiado
import logging
logger = logging.getLogger(__name__)
logger.info("Información importante")
logger.error(f"Error: {e}")

# 3. Manejo de errores robusto
try:
    resultado = herramienta.invoke(args)
except ValidationError as e:
    logger.error(f"Validación fallida: {e}")
    resultado = "Error de validación"
except Exception as e:
    logger.error(f"Error inesperado: {e}")
    resultado = "Ocurrió un error"

# 4. Rate limiting
import time
from google.api_core.exceptions import ResourceExhausted

max_retries = 3
for intento in range(max_retries):
    try:
        response = llm.invoke(prompt)
        break
    except ResourceExhausted:
        if intento < max_retries - 1:
            time.sleep(2 ** intento)  # Backoff exponencial
        else:
            raise

# 5. Validación de inputs
from pydantic import BaseModel, Field, validator

class MiInput(BaseModel):
    email: str
    edad: int = Field(gt=0, lt=150)
    
    @validator('email')
    def validar_email(cls, v):
        if '@' not in v:
            raise ValueError('Email inválido')
        return v

# 6. Documentación de código
def mi_funcion(param1: str, param2: int = 10) -> str:
    """
    Descripción clara de la función.
    
    Args:
        param1: Descripción del parámetro 1
        param2: Descripción del parámetro 2 (default: 10)
    
    Returns:
        Descripción del retorno
    
    Raises:
        ValueError: Si param1 está vacío
    """
```

### ❌ DON'T

```python
# 1. API keys hardcodeadas
api_key = "AIzaSy..."  # ❌

# 2. Sin logging
print("Error")  # ❌ Usa logging

# 3. Sin manejo de errores
resultado = herramienta.invoke(args)  # ❌ Sin try/except

# 4. Sin rate limiting
while True:  # ❌ Sin límites
    response = llm.invoke(prompt)

# 5. Sin validación
def procesar(email: str, edad: int):  # ❌ Sin validar
    pass

# 6. Sin documentación
def f(p1, p2):  # ❌ Sin docstring
    return p1 + p2
```

---

## 📊 Checklist de Producción

### Antes de Desplegar

```
□ Variables de entorno configuradas
□ API keys en .env (no en código)
□ Logging configurado
□ Manejo de errores implementado
□ Rate limiting implementado
□ Validación de inputs
□ Tests unitarios
□ Documentación actualizada
□ Checkpointer persistente (para HITL)
□ Monitoreo configurado (LangSmith)
```

### Seguridad

```
□ No exponer API keys
□ Validar todos los inputs
□ Sanitizar outputs
□ Rate limiting por usuario
□ Autenticación si es necesario
□ HTTPS en producción
□ No loggear información sensible
```

### Performance

```
□ Caching de respuestas
□ Batch processing cuando sea posible
□ Streaming para UX
□ Timeouts configurados
□ Reintentos con backoff
□ Monitoreo de latencia
```

---

## 🎯 Resumen Visual

```
┌────────────────────────────────────────────────────────────┐
│                    MEJORES PRÁCTICAS                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  MÓDULO 1: Fundamentos                                     │
│  ✅ Type hints, docstrings, temperature adecuada           │
│  ❌ Sin validación, sin errores                            │
│                                                            │
│  MÓDULO 2: Memoria                                         │
│  ✅ WindowMemory, SummaryMemory, persistencia              │
│  ❌ Buffer sin límite, no guardar contexto                 │
│                                                            │
│  MÓDULO 3: Streaming                                       │
│  ✅ streaming=True, flush=True, callbacks                  │
│  ❌ Sin flush, mezclar con invoke                          │
│                                                            │
│  MÓDULO 4: LangGraph                                       │
│  ✅ TypedDict, START/END, conditional edges                │
│  ❌ Sin tipo, sin END, sin mapeo                           │
│                                                            │
│  MÓDULO 5: Herramientas                                    │
│  ✅ @tool, Pydantic, snake_case, temperature baja          │
│  ❌ Sin type hints, docstring vago, nombres con espacios   │
│                                                            │
│  MÓDULO 6: HITL                                            │
│  ✅ Checkpointer, contexto claro, thread_id consistente    │
│  ❌ Sin checkpointer, try/except en interrupt              │
│                                                            │
│  PRODUCCIÓN                                                │
│  ✅ .env, logging, errores, rate limiting, validación      │
│  ❌ API keys hardcodeadas, sin logging, sin validación     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

*Guía creada: 2026-03-29*  
*Basado en documentación oficial y experiencia práctica*  
*Próxima actualización: Después del Módulo 7*
