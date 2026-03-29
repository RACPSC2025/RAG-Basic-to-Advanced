# 📚 Ejercicios Resueltos - Módulos 1-6

> **Propósito**: Reforzar conceptos clave con ejercicios prácticos resueltos  
> **Nivel**: Todos los niveles (Básico, Intermedio, Avanzado)  
> **Actualización**: 2026-03-29

---

## 📋 Índice de Ejercicios

### Módulo 1: Fundamentos LangChain
- [Ejercicio 1.1: Prompts Dinámicos](#ejercicio-11-prompts-dinámicos)
- [Ejercicio 1.2: Conversación con Historial](#ejercicio-12-conversación-con-historial)
- [Ejercicio 1.3: Extracción Estructurada](#ejercicio-13-extracción-estructurada)

### Módulo 2: Memoria
- [Ejercicio 2.1: Asistente Personal con Memoria](#ejercicio-21-asistente-personal-con-memoria)
- [Ejercicio 2.2: Persistencia de Conversación](#ejercicio-22-persistencia-de-conversación)

### Módulo 3: Streaming
- [Ejercicio 3.1: Streaming con Progress Bar](#ejercicio-31-streaming-con-progress-bar)

### Módulo 4: LangGraph Básico
- [Ejercicio 4.1: Grafo con Decisiones Múltiples](#ejercicio-41-grafo-con-decisiones-múltiples)
- [Ejercicio 4.2: Grafo con Bucle y Contador](#ejercicio-42-grafo-con-bucle-y-contador)

### Módulo 5: Herramientas
- [Ejercicio 5.1: Calculadora Legal Completa](#ejercicio-51-calculadora-legal-completa)
- [Ejercicio 5.2: Herramienta con Validación](#ejercicio-52-herramienta-con-validación)

### Módulo 6: Human in the Loop
- [Ejercicio 6.1: Aprobación de Demanda](#ejercicio-61-aprobación-de-demanda)
- [Ejercicio 6.2: Revisión de Contrato](#ejercicio-62-revisión-de-contrato)

---

## Módulo 1: Fundamentos LangChain

### Ejercicio 1.1: Prompts Dinámicos

**Enunciado**: Crea un sistema que genere prompts dinámicos para diferentes tipos de documentos legales.

**Solución**:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Template parametrizado
template = ChatPromptTemplate.from_messages([
    ("system", """Eres un experto en derecho {especialidad} colombiano.
    
    Tu tarea: {tarea}
    
    Formato de salida: {formato}
    
    Instrucciones adicionales:
    - Usa lenguaje técnico apropiado
    - Cita normas vigentes
    - Sé preciso en la redacción
    """),
    ("user", "{consulta}")
])

# Función reutilizable
def generar_documento(especialidad, tarea, formato, consulta):
    messages = template.format_messages(
        especialidad=especialidad,
        tarea=tarea,
        formato=formato,
        consulta=consulta
    )
    response = llm.invoke(messages)
    return response.content

# Uso
documento = generar_documento(
    especialidad="laboral",
    tarea="Redactar carta de despido",
    formato="Carta formal con encabezado, cuerpo y cierre",
    consulta="Despido por justa causa: trabajador llegó 5 veces tarde en el mes"
)

print(documento)
```

---

### Ejercicio 1.2: Conversación con Historial

**Enunciado**: Implementa un chatbot legal que mantenga el contexto de la conversación.

**Solución**:

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

class ChatbotLegal:
    def __init__(self, especialidad="colombiano"):
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Últimos 5 turnos
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un asistente legal experto en derecho {especialidad}."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    def conversar(self, mensaje: str) -> str:
        # Cargar historial
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # Formatear prompt
        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=mensaje
        )
        
        # Obtener respuesta
        response = self.llm.invoke(messages)
        
        # Guardar en memoria
        self.memory.save_context(
            {"input": mensaje},
            {"output": response.content}
        )
        
        return response.content
    
    def limpiar(self):
        self.memory.clear()

# Uso
chatbot = ChatbotLegal()

print(chatbot.conversar("¿Qué es una tutela?"))
print(chatbot.conversar("¿Quién puede interponerla?"))  # Mantiene contexto
print(chatbot.conversar("¿Y si soy menor de edad?"))    # Sigue el contexto
```

---

### Ejercicio 1.3: Extracción Estructurada

**Enunciado**: Extrae información de sentencias judiciales en formato estructurado.

**Solución**:

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class SentenciaEstructurada(BaseModel):
    numero: str = Field(description="Número de sentencia")
    fecha: str = Field(description="Fecha de la sentencia (YYYY-MM-DD)")
    corte: str = Field(description="Corte que emitió (CC, CE, CSJ, etc.)")
    magistrado_ponente: str = Field(description="Nombre del magistrado ponente")
    tema: str = Field(description="Tema principal")
    decision: str = Field(description="Decisión final")
    fundamentos: list[str] = Field(description="Fundamentos jurídnicos principales")

# Parser
parser = PydanticOutputParser(pydantic_object=SentenciaEstructurada)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extrae información estructurada de la sentencia judicial."),
    ("user", """Texto de la sentencia:
    {texto}
    
    {format_instructions}
    
    Extrae todos los campos solicitados.""")
])

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)

# Chain
chain = prompt | llm | parser

# Uso
texto_sentencia = """
CORTE CONSTITUCIONAL
SENTENCIA T-123 DE 2024
Magistrado Ponente: Dr. Juan Pérez Martínez
Fecha: 15 de marzo de 2024

Acción de Tutela presentada por María Rodríguez...

DECISIÓN: TUTELA IMPROCEDENTE

FUNDAMENTOS:
1. La acción de tutela no es el mecanismo idóneo
2. Existe otro medio de defensa judicial
3. No se configura un perjuicio irremediable
"""

resultado = chain.invoke({"texto": texto_sentencia})

print(f"Número: {resultado.numero}")
print(f"Corte: {resultado.corte}")
print(f"Decisión: {resultado.decision}")
print(f"Fundamentos: {len(resultado.fundamentos)} encontrados")
```

---

## Módulo 2: Memoria

### Ejercicio 2.1: Asistente Personal con Memoria

**Enunciado**: Crea un asistente que recuerde preferencias del usuario entre sesiones.

**Solución**:

```python
import json
from pathlib import Path
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

class AsistentePersonal:
    def __init__(self, usuario_id: str):
        self.usuario_id = usuario_id
        self.archivo_perfil = Path(f"perfiles/{usuario_id}.json")
        
        # Cargar perfil existente
        self.perfil = self.cargar_perfil()
        
        # Memoria de resumen
        self.memory = ConversationSummaryMemory(
            llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp"),
            memory_key="resumen",
            return_messages=False
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente personal.
            
            Información del usuario:
            {perfil}
            
            Resumen de conversaciones anteriores:
            {resumen}
            
            Sé útil y personalizado en tus respuestas."""),
            ("user", "{input}")
        ])
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    def cargar_perfil(self):
        if self.archivo_perfil.exists():
            with open(self.archivo_perfil, 'r') as f:
                return json.load(f)
        return {"nombre": "Usuario", "preferencias": {}, "historial": []}
    
    def guardar_perfil(self):
        self.archivo_perfil.parent.mkdir(exist_ok=True)
        with open(self.archivo_perfil, 'w') as f:
            json.dump(self.perfil, f, indent=2)
    
    def conversar(self, mensaje: str) -> str:
        # Obtener resumen de memoria
        resumen = self.memory.load_memory_variables({}).get("resumen", "")
        
        # Generar respuesta
        messages = self.prompt.format_messages(
            perfil=json.dumps(self.perfil, indent=2),
            resumen=resumen,
            input=mensaje
        )
        
        response = self.llm.invoke(messages)
        
        # Actualizar memoria
        self.memory.save_context(
            {"input": mensaje},
            {"output": response.content}
        )
        
        # Actualizar historial
        self.perfil["historial"].append({
            "input": mensaje,
            "output": response.content
        })
        self.guardar_perfil()
        
        return response.content
    
    def actualizar_preferencia(self, clave: str, valor: str):
        self.perfil["preferencias"][clave] = valor
        self.guardar_perfil()
        return f"Preferencia '{clave}' guardada"

# Uso
asistente = AsistentePersonal("usuario_123")

print(asistente.conversar("Me llamo Carlos"))
print(asistente.conversar("Trabajo como abogado"))
print(asistente.conversar("Prefiero el derecho laboral"))

# Nueva sesión (mantiene información)
asistente2 = AsistentePersonal("usuario_123")
print(asistente2.conversar("¿Cómo me llamo?"))  # Recuerda: Carlos
```

---

## Módulo 3: Streaming

### Ejercicio 3.1: Streaming con Progress Bar

**Enunciado**: Implementa streaming con barra de progreso para procesos largos.

**Solución**:

```python
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class StreamingConProgreso:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            streaming=True
        )
    
    def generar_con_progreso(self, prompt: str):
        template = ChatPromptTemplate.from_messages([
            ("user", "{prompt}")
        ])
        
        chain = template | self.llm
        
        print("Generando respuesta...")
        print("=" * 60)
        
        contador = 0
        for chunk in chain.stream({"prompt": prompt}):
            print(chunk.content, end="", flush=True)
            contador += 1
            
            # Actualizar progress bar cada 10 tokens
            if contador % 10 == 0:
                progreso = min(contador / 100, 1.0) * 100
                print(f"\n[{'█' * int(progreso/5)}{'░' * (20 - int(progreso/5))}] {progreso:.0f}%")
        
        print("\n" + "=" * 60)
        print(f"✅ Generación completada ({contador} tokens)")

# Uso
generator = StreamingConProgreso()
generator.generar_con_progreso(
    "Explica detalladamente el proceso de la acción de tutela en Colombia, "
    "incluyendo todos los pasos desde la presentación hasta la notificación de la sentencia."
)
```

---

## Módulo 4: LangGraph Básico

### Ejercicio 4.1: Grafo con Decisiones Múltiples

**Enunciado**: Crea un clasificador de documentos legales que route a diferentes nodos.

**Solución**:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

class DocumentoState(TypedDict):
    texto: str
    tipo: str
    analisis: str

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

def clasificar(state: DocumentoState):
    """Clasifica el tipo de documento."""
    prompt = f"""Clasifica este documento legal:
    {state['texto'][:200]}...
    
    Tipos posibles: contrato, demanda, sentencia, ley, otro
    
    Responde SOLO con el tipo:"""
    
    response = llm.invoke(prompt)
    return {"tipo": response.content.strip().lower()}

def analizar_contrato(state: DocumentoState):
    return {"analisis": "📄 Analizando contrato..."}

def analizar_demanda(state: DocumentoState):
    return {"analisis": "⚖️ Analizando demanda..."}

def analizar_sentencia(state: DocumentoState):
    return {"analisis": "📋 Analizando sentencia..."}

def analizar_generico(state: DocumentoState):
    return {"analisis": "📝 Analizando documento genérico..."}

# Grafo
builder = StateGraph(DocumentoState)

# Nodos
builder.add_node("clasificar", clasificar)
builder.add_node("analizar_contrato", analizar_contrato)
builder.add_node("analizar_demanda", analizar_demanda)
builder.add_node("analizar_sentencia", analizar_sentencia)
builder.add_node("analizar_generico", analizar_generico)

# Edges
builder.add_edge(START, "clasificar")

def router(state: DocumentoState) -> str:
    tipo = state["tipo"]
    if "contrato" in tipo:
        return "analizar_contrato"
    elif "demanda" in tipo:
        return "analizar_demanda"
    elif "sentencia" in tipo:
        return "analizar_sentencia"
    else:
        return "analizar_generico"

builder.add_conditional_edges("clasificar", router, {
    "analizar_contrato": "analizar_contrato",
    "analizar_demanda": "analizar_demanda",
    "analizar_sentencia": "analizar_sentencia",
    "analizar_generico": "analizar_generico"
})

builder.add_edge("analizar_contrato", END)
builder.add_edge("analizar_demanda", END)
builder.add_edge("analizar_sentencia", END)
builder.add_edge("analizar_generico", END)

graph = builder.compile()

# Uso
resultado = graph.invoke({
    "texto": "CONTRATO DE PRESTACIÓN DE SERVICIOS...",
    "tipo": "",
    "analisis": ""
})

print(f"Tipo: {resultado['tipo']}")
print(f"Análisis: {resultado['analisis']}")
```

---

## Módulo 5: Herramientas

### Ejercicio 5.1: Calculadora Legal Completa

**Enunciado**: Crea herramientas para calcular intereses, fechas y términos procesales.

**Solución**:

```python
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain.tools import tool

# Tool 1: Calcular intereses moratorios
@tool
def calcular_intereses_moratorios(
    capital: float,
    dias_mora: int,
    tasa_anual: float = 0.24
) -> dict:
    """
    Calcula intereses moratorios según la tasa certificada.
    
    Args:
        capital: Capital base
        dias_mora: Días de mora
        tasa_anual: Tasa anual (default: 24% máxima permitida)
    
    Returns:
        Diccionario con el desglose del cálculo
    """
    tasa_diaria = tasa_anual / 360
    interes = capital * tasa_diaria * dias_mora
    total = capital + interes
    
    return {
        "capital": capital,
        "dias_mora": dias_mora,
        "tasa_anual": f"{tasa_anual*100}%",
        "tasa_diaria": f"{tasa_diaria*100:.4f}%",
        "interes_calculado": f"${interes:,.0f}",
        "total_a_pagar": f"${total:,.0f}"
    }

# Tool 2: Calcular fecha de vencimiento
@tool
def calcular_vencimiento(
    fecha_inicial: str,
    termino_dias: int,
    tipo_dias: str = "habiles"
) -> dict:
    """
    Calcula fecha de vencimiento de términos procesales.
    
    Args:
        fecha_inicial: Fecha inicial (YYYY-MM-DD)
        termino_dias: Número de días del término
        tipo_dias: 'habiles' o 'calendario'
    
    Returns:
        Fechas calculadas
    """
    inicio = datetime.strptime(fecha_inicial, "%Y-%m-%d")
    
    if tipo_dias == "habiles":
        # Simplificación: excluir sábados y domingos
        dias_contados = 0
        fecha_actual = inicio
        while dias_contados < termino_dias:
            fecha_actual += timedelta(days=1)
            if fecha_actual.weekday() < 5:  # Lunes a viernes
                dias_contados += 1
        vencimiento = fecha_actual
    else:
        vencimiento = inicio + timedelta(days=termino_dias)
    
    return {
        "fecha_inicial": fecha_inicial,
        "termino": f"{termino_dias} días {tipo_dias}",
        "fecha_vencimiento": vencimiento.strftime("%Y-%m-%d"),
        "dias_calendario": (vencimiento - inicio).days
    }

# Tool 3: Verificar términos por tipo de proceso
@tool
def consultar_terminos_procesales(tipo_proceso: str) -> str:
    """
    Consulta términos procesales por tipo de proceso.
    
    Args:
        tipo_proceso: Tipo de proceso (civil, laboral, penal, administrativo)
    
    Returns:
        Términos aplicables
    """
    terminos = {
        "civil": {
            "demanda": "10 días para contestar",
            "apelacion": "5 días para apelar",
            "ejecucion": "3 días para pagar"
        },
        "laboral": {
            "demanda": "10 días para contestar",
            "apelacion": "3 días para apelar",
            "liquidacion": "5 días para objetar"
        },
        "penal": {
            "contestacion": "30 días",
            "pruebas": "90 días",
            "alegatos": "10 días"
        },
        "administrativo": {
            "demanda": "30 días para contestar",
            "apelacion": "10 días para apelar"
        }
    }
    
    if tipo_proceso.lower() in terminos:
        resultado = f"Términos para proceso {tipo_proceso.capitalize()}:\n\n"
        for concepto, termino in terminos[tipo_proceso.lower()].items():
            resultado += f"• {concepto.capitalize()}: {termino}\n"
        return resultado
    else:
        return f"Tipo de proceso '{tipo_proceso}' no reconocido"

# Uso
print(calcular_intereses_moratorios.invoke({
    "capital": 10000000,
    "dias_mora": 90,
    "tasa_anual": 0.24
}))

print(calcular_vencimiento.invoke({
    "fecha_inicial": "2024-01-15",
    "termino_dias": 10,
    "tipo_dias": "habiles"
}))

print(consultar_terminos_procesales.invoke({
    "tipo_proceso": "laboral"
}))
```

---

## Módulo 6: Human in the Loop

### Ejercicio 6.1: Aprobación de Demanda

**Enunciado**: Crea un flujo que requiera aprobación de abogado antes de presentar demanda.

**Solución**:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI

class DemandaState(TypedDict):
    tipo_accion: str
    hechos: str
    pretensiones: str
    documentos: list[str]
    estado: Literal["borrador", "revision", "aprobado", "rechazado"]
    observaciones: str

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

def generar_borrador(state: DemandaState):
    """Genera borrador de demanda."""
    prompt = f"""
    Genera el borrador de una {state['tipo_accion']} con:
    
    Hechos: {state['hechos']}
    Pretensiones: {state['pretensiones']}
    Documentos: {', '.join(state['documentos'])}
    
    Incluye:
    1. Encabezado
    2. Hechos
    3. Pretensiones
    4. Fundamentos de derecho
    5. Pruebas
    """
    
    response = llm.invoke(prompt)
    return {"estado": "revision", "observaciones": response.content}

def revision_abogado(state: DemandaState) -> Command[Literal["presentar", "corregir"]]:
    """Pausa para aprobación del abogado."""
    
    decision = interrupt({
        "tipo": "aprobacion_demanda",
        "pregunta": "¿Presentar esta demanda?",
        "demanda": {
            "tipo": state["tipo_accion"],
            "borrador_preview": state["observaciones"][:300] + "...",
            "documentos_anexos": state["documentos"]
        },
        "checklist": [
            "✓ Hechos claros y completos",
            "✓ Pretensiones bien formuladas",
            "✓ Pruebas documentales anexadas",
            "✓ Fundamentos jurídnicos correctos",
            "✓ Competencia del juez verificada"
        ]
    })
    
    if decision.get("aprobar"):
        return Command(goto="presentar")
    else:
        return Command(goto="corregir")

def presentar_demanda(state: DemandaState):
    """Presenta la demanda."""
    return {"estado": "aprobado", "observaciones": "✅ Demanda presentada"}

def corregir_demanda(state: DemandaState):
    """Corrige la demanda."""
    return {"estado": "borrador", "observaciones": "❌ Demanda devuelta para corrección"}

# Grafo
builder = StateGraph(DemandaState)
builder.add_node("generar", generar_borrador)
builder.add_node("revision", revision_abogado)
builder.add_node("presentar", presentar_demanda)
builder.add_node("corregir", corregir_demanda)

builder.add_edge(START, "generar")
builder.add_edge("generar", "revision")
builder.add_edge("presentar", END)
builder.add_edge("corregir", END)

graph = builder.compile(checkpointer=MemorySaver())

# Uso
config = {"configurable": {"thread_id": "demanda-001"}}

resultado = graph.invoke({
    "tipo_accion": "Demanda Laboral Ordinaria",
    "hechos": "Despido injustificado el 15 de enero de 2024...",
    "pretensiones": "Pago de indemnización, cesantías, intereses...",
    "documentos": ["Contrato", "Carta de despido", "Nóminas"],
    "estado": "borrador",
    "observaciones": ""
}, config)

# Reanudar con decisión
# graph.invoke(Command(resume={"aprobar": True}), config)
```

---

## 📝 Conclusión

Estos ejercicios refuerzan los conceptos clave de cada módulo. Se recomienda:

1. **Ejecutar cada ejercicio** para entender el funcionamiento
2. **Modificar parámetros** para ver cómo cambian los resultados
3. **Combinar ejercicios** para crear soluciones más complejas
4. **Adaptar al caso de uso** específico de cada estudiante

---

*Ejercicios creados: 2026-03-29*  
*Próxima actualización: Más ejercicios de RAG*
