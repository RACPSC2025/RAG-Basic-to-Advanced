"""
Módulo 5.1 - Creación Básica de Herramientas

Objetivo: Aprender a crear herramientas con el decorador @tool
Basado en: https://docs.langchain.com/oss/python/langchain/tools
"""

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar variables de entorno
load_dotenv()


# ============================================================================
# EJEMPLO 1: Herramienta Básica
# ============================================================================

@tool
def buscar_ley(nombre: str) -> str:
    """
    Busca información sobre una ley o mecanismo legal colombiano.
    
    Args:
        nombre: Nombre de la ley (ej: 'tutela', 'derecho_peticion', 'habeas_corpus')
    
    Returns:
        Información completa de la ley: nombre, descripción, base legal y tiempo de respuesta
    """
    
    leyes = {
        "tutela": {
            "nombre": "Acción de Tutela",
            "descripcion": "Mecanismo constitucional para proteger derechos fundamentales",
            "articulo": "Artículo 86 de la Constitución Política",
            "tiempo_respuesta": "10 días hábiles"
        },
        "derecho_peticion": {
            "nombre": "Derecho de Petición",
            "descripcion": "Derecho fundamental para solicitar información a autoridades",
            "articulo": "Artículo 23 de la Constitución Política",
            "tiempo_respuesta": "15 días hábiles"
        },
        "habeas_corpus": {
            "nombre": "Habeas Corpus",
            "descripcion": "Mecanismo para proteger la libertad personal",
            "articulo": "Artículo 30 de la Constitución Política",
            "tiempo_respuesta": "36 horas"
        }
    }
    
    # Normalizar nombre
    nombre_key = nombre.lower().strip().replace(" ", "_")
    
    if nombre_key in leyes:
        ley = leyes[nombre_key]
        return f"""
        ╔═══════════════════════════════════════════════════════════╗
        Ley: {ley['nombre']}
        ╠═══════════════════════════════════════════════════════════╣
        Descripción: {ley['descripcion']}
        Base Legal: {ley['articulo']}
        Tiempo de respuesta: {ley['tiempo_respuesta']}
        ╚═══════════════════════════════════════════════════════════╝
        """
    else:
        opciones = ", ".join(leyes.keys())
        return f"❌ Ley '{nombre}' no encontrada.\n\nOpciones disponibles: {opciones}"


# ============================================================================
# EJEMPLO 2: Herramienta con Nombre Personalizado
# ============================================================================

@tool("calcular_fecha_procesal",
      description="Calcula fechas en procesos judiciales colombianos. Úsala para determinar vencimientos de términos procesales.")
def calcular_fecha(dias: int, tipo: str = "habiles") -> str:
    """
    Calcula fechas procesales sumando días a la fecha actual.
    
    Args:
        dias: Número de días a sumar
        tipo: Tipo de días ('habiles' o 'calendario')
    
    Returns:
        Fecha inicial y fecha resultante
    """
    from datetime import datetime, timedelta
    
    hoy = datetime.now()
    
    # En producción, para días hábiles excluir fines de semana y festivos
    if tipo.lower() == "habiles":
        # Simplificación: suma días (en producción, excluir fines de semana)
        resultado = hoy + timedelta(days=dias)
    else:
        resultado = hoy + timedelta(days=dias)
    
    return f"""
    📅 Cálculo de Fecha Procesal
    ════════════════════════════
    Fecha inicial: {hoy.strftime('%Y-%m-%d')}
    Días a sumar: {dias} ({tipo})
    Fecha resultante: {resultado.strftime('%Y-%m-%d')}
    """


# ============================================================================
# EJEMPLO 3: Múltiples Herramientas con Agente
# ============================================================================

def crear_agente_legal():
    """
    Crea un agente con múltiples herramientas legales.
    
    Returns:
        Agente configurado con todas las herramientas
    """
    
    # Inicializar LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,  # Más determinista para herramientas
    )
    
    # Crear agente con herramientas
    agent = create_agent(
        llm,
        tools=[buscar_ley, calcular_fecha],
        system_prompt="""Eres un asistente legal experto en derecho colombiano.
        
        Tus capacidades:
        - Buscar información sobre leyes y mecanismos constitucionales
        - Calcular fechas procesales
        
        Instrucciones:
        - Usa las herramientas para obtener información precisa
        - Responde en español de manera clara y profesional
        - Si no sabes algo, admítelo honestamente
        - No des consejos legales vinculantes
        """
    )
    
    return agent


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def main():
    """Función principal para demostrar el uso de herramientas"""
    
    print("=" * 80)
    print("MÓDULO 5.1 - CREACIÓN BÁSICA DE HERRAMIENTAS")
    print("=" * 80)
    
    # 1. Mostrar propiedades de la herramienta
    print("\n1️⃣ PROPIEDADES DE LA HERRAMIENTA")
    print("-" * 80)
    print(f"Nombre: {buscar_ley.name}")
    print(f"Descripción: {buscar_ley.description}")
    print(f"Schema de argumentos: {buscar_ley.args}")
    
    # 2. Probar herramienta directamente
    print("\n2️⃣ PRUEBA DIRECTA DE HERRAMIENTA")
    print("-" * 80)
    
    resultados = [
        buscar_ley.invoke({"nombre": "tutela"}),
        buscar_ley.invoke({"nombre": "derecho_peticion"}),
        calcular_fecha.invoke({"dias": 10, "tipo": "habiles"}),
    ]
    
    for resultado in resultados:
        print(f"\n{resultado}")
    
    # 3. Usar con agente
    print("\n3️⃣ USAR HERRAMIENTAS CON AGENTE")
    print("-" * 80)
    
    agent = crear_agente_legal()
    
    preguntas = [
        "¿Qué es una tutela y cuánto tiempo tienen para responder?",
        "Si hoy es el día 1 de un proceso y tengo 10 días hábiles, cuándo vence el término?",
        "Compara la tutela con el habeas corpus",
    ]
    
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n{'='*80}")
        print(f"PREGUNTA {i}: {pregunta}")
        print(f"{'='*80}")
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": pregunta}]
        })
        
        print(f"\nRESPUESTA DEL AGENTE:\n{response['messages'][-1].content}")


# ============================================================================
# MEJORES PRÁCTICAS
# ============================================================================

"""
✅ DO (Haz esto):

1. Siempre usa type hints
   @tool
   def mi_funcion(param: str) -> str:  # ✅

2. Docstring descriptivo
   '''Busca información específica sobre X tema.
   
   Args:
       param: Descripción del parámetro
   
   Returns:
       Descripción del retorno
   '''

3. Nombres en snake_case
   @tool("buscar_ley")  # ✅

4. Temperature baja para herramientas
   ChatGoogleGenerativeAI(temperature=0.3)  # ✅


❌ DON'T (No hagas esto):

1. Sin type hints
   def mi_funcion(param):  # ❌

2. Docstring vago
   '''Busca algo'''  # ❌

3. Nombres con espacios
   @tool("Buscar Ley")  # ❌

4. Temperature alta
   ChatGoogleGenerativeAI(temperature=1.5)  # ❌
"""


if __name__ == "__main__":
    main()
