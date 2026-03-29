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
# 3. SYSTEM PROMPT vs USER PROMPT
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
# 4. TEMPLATE CON FUNCIONES
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
    system_vs_user_prompt()
    template_con_funciones()
