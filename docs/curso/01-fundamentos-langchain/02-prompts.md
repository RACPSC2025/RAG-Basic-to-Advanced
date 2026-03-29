# Módulo 1.2: Prompts - Entrada y Estructura

## Objetivos
- Comprender qué es un prompt y su estructura
- Usar PromptTemplate para crear prompts dinámicos
- Implementar few-shot prompting
- Diferenciar system prompts de user prompts

---

## 1.2.1 ¿Qué es un Prompt?

Un **prompt** es la entrada de texto que le das al LLM para obtener una respuesta.

### Estructura Básica de un Prompt

```
┌─────────────────────────────────────────────────────────┐
│                    SYSTEM PROMPT                        │
│  (Instrucciones, rol, contexto, comportamiento)         │
├─────────────────────────────────────────────────────────┤
│                     USER PROMPT                         │
│  (Pregunta, tarea, información específica)              │
├─────────────────────────────────────────────────────────┤
│                   FEW-SHOT EXAMPLES                     │
│  (Opcional: ejemplos de entrada/salida)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 1.2.2 PromptTemplate Básico

### Código de Ejemplo

Archivo: `src/course_examples/modulo_01/02_prompts.py`

```python
"""
02_prompts.py
Trabajando con Prompts en LangChain

Objetivo: Dominar PromptTemplate y few-shot prompting
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
)


# ============================================================
# 1. PROMPT TEMPLATE BÁSICO
# ============================================================

def prompt_template_basico():
    """Crear y usar un PromptTemplate básico"""
    
    # Forma INCORRECTA (string fijo)
    prompt_fijo = "Traduce al inglés: Hola mundo"
    
    # Forma CORRECTA (template con variables)
    template = "Traduce al {idioma_destino}: {texto}"
    
    # Crear el template
    prompt = PromptTemplate(
        template=template,
        input_variables=["idioma_destino", "texto"],
    )
    
    # Usar el template
    prompt_completo = prompt.format(
        idioma_destino="inglés",
        texto="Hola mundo"
    )
    
    print("=" * 60)
    print("PROMPT TEMPLATE BÁSICO")
    print("=" * 60)
    print(f"Template: {template}")
    print(f"Formato: {prompt_completo}")
    
    response = llm.invoke(prompt_completo)
    print(f"Respuesta: {response.content}\n")
    
    # Probar con diferentes idiomas
    for idioma in ["francés", "alemán", "italiano"]:
        prompt_completo = prompt.format(
            idioma_destino=idioma,
            texto="Buenos días"
        )
        response = llm.invoke(prompt_completo)
        print(f"{idioma}: {response.content}")


# ============================================================
# 2. CHAT PROMPT TEMPLATE (RECOMENDADO)
# ============================================================

def chat_prompt_template():
    """Usar ChatPromptTemplate para conversaciones"""
    
    # ChatPromptTemplate maneja roles (system, user, assistant)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente legal experto en derecho {pais}."),
        ("user", "{pregunta}"),
    ])
    
    print("\n" + "=" * 60)
    print("CHAT PROMPT TEMPLATE")
    print("=" * 60)
    
    # Crear el prompt completo
    messages = chat_prompt.format_messages(
        pais="Colombia",
        pregunta="¿Qué es una acción de tutela?"
    )
    
    print(f"Messages: {messages}")
    
    # Invocar el modelo
    response = llm.invoke(messages)
    print(f"Respuesta: {response.content}\n")
    
    # Reutilizar con diferentes parámetros
    messages = chat_prompt.format_messages(
        pais="México",
        pregunta="¿Qué es un juicio de amparo?"
    )
    
    response = llm.invoke(messages)
    print(f"México - Amparo: {response.content}\n")


# ============================================================
# 3. FEW-SHOT PROMPTING
# ============================================================

def few_shot_prompting():
    """Incluir ejemplos en el prompt para guiar al modelo"""
    
    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
    
    # Ejemplos de entrada/salida
    ejemplos = [
        {
            "pregunta": "¿Qué es una tutela?",
            "respuesta": "Mecanismo constitucional para proteger derechos fundamentales."
        },
        {
            "pregunta": "¿Qué es una demanda?",
            "respuesta": "Acto procesal que inicia un juicio."
        },
        {
            "pregunta": "¿Qué es un contrato?",
            "respuesta": "Acuerdo de voluntades que crea obligaciones."
        },
    ]
    
    # Template para cada ejemplo
    ejemplo_template = PromptTemplate(
        input_variables=["pregunta", "respuesta"],
        template="P: {pregunta}\nR: {respuesta}"
    )
    
    # Template para el prefijo (instrucciones)
    prefijo = """Eres un asistente legal. Responde de forma breve y precisa.

Ejemplos:
{ejemplos}

Ahora responde la siguiente pregunta:"""
    
    # Template para el sufijo (la pregunta actual)
    sufijo = "P: {pregunta_actual}\nR:"
    
    # Crear FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        examples=ejectos,
        example_prompt=ejecto_template,
        input_variables=["pregunta_actual"],
        prefix=prefijo,
        suffix=sufijo,
    )
    
    print("=" * 60)
    print("FEW-SHOT PROMPTING")
    print("=" * 60)
    
    # Generar prompt completo
    prompt_completo = few_shot_prompt.format(
        pregunta_actual="¿Qué es un habeas corpus?"
    )
    
    print(f"Prompt completo:\n{prompt_completo}\n")
    
    response = llm.invoke(prompt_completo)
    print(f"Respuesta: {response.content}\n")


# ============================================================
# 4. SYSTEM PROMPT vs USER PROMPT
# ============================================================

def system_vs_user_prompt():
    """Diferenciar system prompts de user prompts"""
    
    # System prompt: Define el comportamiento del asistente
    system_prompt = """Eres un asistente legal especializado en derecho laboral colombiano.
    
Tu comportamiento:
- Responde de forma clara y concisa
- Cita artículos de la ley cuando sea relevante
- Si no sabes algo, admítelo
- No des consejos legales vinculantes
- Usa lenguaje profesional pero accesible"""
    
    # User prompt: La pregunta específica del usuario
    user_prompt = "¿Cuántos días de vacaciones me corresponden si trabajé 1 año?"
    
    # Combinar ambos en ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    
    print("=" * 60)
    print("SYSTEM PROMPT vs USER PROMPT")
    print("=" * 60)
    
    messages = chat_prompt.format_messages()
    response = llm.invoke(messages)
    
    print(f"Pregunta: {user_prompt}")
    print(f"Respuesta: {response.content}\n")
    
    # El system prompt PERSISTE en la conversación
    # Puedes hacer follow-up questions
    follow_up = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
        ("assistant", response.content),
        ("user", "¿Y si trabajé solo 6 meses?"),
    ])
    
    messages = follow_up.format_messages()
    response = llm.invoke(messages)
    
    print(f"Follow-up: ¿Y si trabajé solo 6 meses?")
    print(f"Respuesta: {response.content}\n")


# ============================================================
# 5. TEMPLATE CON FUNCIONES
# ============================================================

def template_con_funciones():
    """Crear funciones reutilizables con templates"""
    
    # Template para resumir documentos legales
    resumir_template = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en analizar documentos legales."),
        ("user", """Resume el siguiente documento en {num_puntos} puntos clave.
        Enfócate en los aspectos más relevantes para un abogado.
        
        Documento:
        {documento}
        
        Resumen:"""),
    ])
    
    # Función reutilizable
    def resumir_documento(documento: str, num_puntos: int = 5):
        messages = resumir_template.format_messages(
            documento=documento[:1000],  # Limitar longitud
            num_puntos=num_puntos
        )
        response = llm.invoke(messages)
        return response.content
    
    print("=" * 60)
    print("TEMPLATE CON FUNCIONES")
    print("=" * 60)
    
    documento_ejemplo = """
    LEY 1564 DE 2012
    Código General del Proceso
    
    Artículo 1. Objeto. El presente código tiene por objeto establecer las reglas 
    de procedimiento para los procesos judiciales que se adelanten ante la jurisdicción 
    ordinaria en los órdenes civil, laboral y agrario.
    
    Artículo 2. Principios fundamentales. La actuación procesal se regirá por los 
    principios de celeridad, economía, eficacia y transparencia.
    """
    
    resumen = resumir_documento(documento_ejemplo, num_puntos=3)
    print(f"Resumen:\n{resumen}\n")


if __name__ == "__main__":
    prompt_template_basico()
    chat_prompt_template()
    few_shot_prompting()
    system_vs_user_prompt()
    template_con_funciones()

```

---

## 1.2.3 Tipos de Prompts

### 1. PromptTemplate (Básico)

Para prompts simples con variables:

```python
from langchain_core.prompts import PromptTemplate

template = "Explica {concepto} como si tuviera {edad} años"
prompt = PromptTemplate(template=template, input_variables=["concepto", "edad"])

# Usar
prompt.format(concepto="blockchain", edad=10)
# Output: "Explica blockchain como si tuviera 10 años"
```

### 2. ChatPromptTemplate (Recomendado)

Para conversaciones con roles:

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un {rol} experto en {area}."),
    ("user", "{pregunta}"),
])

# Usar
messages = chat_prompt.format_messages(
    rol="abogado",
    area="derecho laboral",
    pregunta="¿Qué es el despido injustificado?"
)
```

### 3. FewShotPromptTemplate

Para incluir ejemplos:

```python
from langchain_core.prompts import FewShotPromptTemplate

ejemplos = [
    {"input": "Hola", "output": "¡Buenos días!"},
    {"input": "Adiós", "output": "¡Hasta luego!"},
]

few_shot = FewShotPromptTemplate(
    examples=ejemplos,
    input_variables=["input"],
)
```

---

## 1.2.4 Mejores Prácticas para Prompts

### ✅ DO (Haz esto)

```python
# 1. Sé específico en el system prompt
system = """Eres un asistente legal colombiano.
- Cita artículos específicos
- Usa lenguaje técnico pero claro
- Máximo 3 párrafos por respuesta"""

# 2. Usa variables para reutilizar templates
template = ChatPromptTemplate.from_messages([
    ("system", "Eres experto en {area}"),
    ("user", "{pregunta}"),
])

# 3. Incluye contexto relevante
prompt = """
Contexto: {contexto_legal}

Pregunta: {pregunta}

Responde basándote ÚNICAMENTE en el contexto proporcionado.
"""

# 4. Especifica el formato de salida
prompt = """
Resume este documento legal.

Formato de salida:
1. Punto clave 1
2. Punto clave 2
3. Punto clave 3

Documento: {documento}
"""
```

### ❌ DON'T (No hagas esto)

```python
# 1. Prompts vagos
prompt = "Habla de leyes"  # ❌ Muy genérico

# 2. Múltiples tareas en un prompt
prompt = """
Resume este documento, tradúcelo al inglés, 
y dame tu opinión personal  # ❌ Demasiado
"""

# 3. Sin contexto
prompt = "¿Es legal esto?"  # ❌ ¿Qué es "esto"?

# 4. Esperar formato específico sin instruir
prompt = "Dame los artículos"  # ❌ ¿En qué formato?
```

---

## 1.2.5 Ejercicios Prácticos

### Ejercicio 1: Template de Traducción

Crea un `PromptTemplate` que:
- Tenga variables para idioma origen, idioma destino y texto
- Prueba traduciendo de español a 3 idiomas diferentes

### Ejercicio 2: Asistente con Personalidad

Crea un `ChatPromptTemplate` para un asistente que:
- Tenga una personalidad específica (ej: formal, amigable, sarcástico)
- Responda preguntas sobre un tema específico
- Prueba con 3 personalidades diferentes

### Ejercicio 3: Few-Shot para Clasificación

Crea un `FewShotPromptTemplate` que:
- Tenga 5 ejemplos de clasificación de documentos legales
- Clasifique un nuevo documento como: "Civil", "Penal", "Laboral", "Administrativo"

---

## 1.2.6 Recursos Adicionales

### Documentación Oficial
- [LangChain Prompt Templates](https://docs.langchain.com/oss/python/langchain/concepts/prompt_templates)
- [LangChain Chat Prompt](https://docs.langchain.com/oss/python/langchain/concepts/chat_models)
- [Few-Shot Learning](https://docs.langchain.com/oss/python/langchain/concepts/few_shot_prompting)

### Siguiente Lección
➡️ **1.3 Mensajes y Chat Models**

---

*Lección creada: 2026-03-29*
