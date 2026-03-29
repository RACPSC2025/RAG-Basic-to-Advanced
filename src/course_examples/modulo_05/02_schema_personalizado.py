"""
Módulo 5.2 - Herramientas con Schema Personalizado

Objetivo: Aprender a crear herramientas con schemas complejos usando Pydantic
Basado en: https://docs.langchain.com/oss/python/langchain/tools
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar variables de entorno
load_dotenv()


# ============================================================================
# EJEMPLO 1: Schema con Múltiples Argumentos
# ============================================================================

class CompararLeyesInput(BaseModel):
    """Schema para comparar dos leyes colombianas."""
    
    ley1: str = Field(
        description="Primera ley o mecanismo a comparar (ej: 'tutela', 'habeas_corpus')"
    )
    ley2: str = Field(
        description="Segunda ley o mecanismo a comparar"
    )
    incluir_tiempos: bool = Field(
        default=True,
        description="Si incluir comparación de tiempos de respuesta en el resultado"
    )
    incluir_articulos: bool = Field(
        default=True,
        description="Si incluir artículos constitucionales en el resultado"
    )


@tool(args_schema=CompararLeyesInput)
def comparar_leyes(
    ley1: str,
    ley2: str,
    incluir_tiempos: bool = True,
    incluir_articulos: bool = True
) -> str:
    """
    Compara dos leyes o mecanismos constitucionales colombianos.
    
    Proporciona un análisis detallado de similitudes y diferencias entre
    dos mecanismos legales, incluyendo tiempos de respuesta y bases legales.
    
    Args:
        ley1: Primer mecanismo legal a comparar
        ley2: Segundo mecanismo legal a comparar
        incluir_tiempos: Si incluir comparación de tiempos
        incluir_articulos: Si incluir artículos constitucionales
    
    Returns:
        Comparación detallada de las leyes
    """
    
    leyes = {
        "tutela": {
            "nombre": "Acción de Tutela",
            "tipo": "Mecanismo de protección inmediata",
            "articulo": "Artículo 86 CP",
            "tiempo": "10 días hábiles",
            "procede": "Derechos fundamentales",
            "termino": "Inmediato"
        },
        "derecho_peticion": {
            "nombre": "Derecho de Petición",
            "tipo": "Derecho fundamental",
            "articulo": "Artículo 23 CP",
            "tiempo": "15 días hábiles",
            "procede": "Cualquier solicitud",
            "termino": "General"
        },
        "habeas_corpus": {
            "nombre": "Habeas Corpus",
            "tipo": "Mecanismo de protección",
            "articulo": "Artículo 30 CP",
            "tiempo": "36 horas",
            "procede": "Libertad personal",
            "termino": "Urgente"
        },
        "accion_popular": {
            "nombre": "Acción Popular",
            "tipo": "Mecanismo de protección colectiva",
            "articulo": "Artículo 88 CP",
            "tiempo": "Según complejidad",
            "procede": "Derechos colectivos",
            "termino": "Ordinario"
        }
    }
    
    # Validar que las leyes existen
    ley1_key = ley1.lower().replace(" ", "_")
    ley2_key = ley2.lower().replace(" ", "_")
    
    if ley1_key not in leyes:
        return f"❌ Ley '{ley1}' no encontrada. Opciones: {', '.join(leyes.keys())}"
    
    if ley2_key not in leyes:
        return f"❌ Ley '{ley2}' no encontrada. Opciones: {', '.join(leyes.keys())}"
    
    l1 = leyes[ley1_key]
    l2 = leyes[ley2_key]
    
    # Construir comparación
    resultado = f"""
╔═══════════════════════════════════════════════════════════════════════╗
                    COMPARACIÓN DE MECANISMOS LEGALES
╚═══════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────┐
│ MECANISMO 1: {l1['nombre']:<54} │
├───────────────────────────────────────────────────────────────────────┤
│ Tipo: {l1['tipo']:<54} │
│ Procedencia: {l1['procede']:<54} │
│ Término: {l1['termino']:<54} │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ MECANISMO 2: {l2['nombre']:<54} │
├───────────────────────────────────────────────────────────────────────┤
│ Tipo: {l2['tipo']:<54} │
│ Procedencia: {l2['procede']:<54} │
│ Término: {l2['termino']:<54} │
└───────────────────────────────────────────────────────────────────────┘
"""
    
    # Agregar tiempos si se solicita
    if incluir_tiempos:
        resultado += f"""
┌───────────────────────────────────────────────────────────────────────┐
│ COMPARACIÓN DE TIEMPOS DE RESPUESTA                                   │
├───────────────────────────────────────────────────────────────────────┤
│ {l1['nombre']:<20} → {l1['tiempo']:<35} │
│ {l2['nombre']:<20} → {l2['tiempo']:<35} │
└───────────────────────────────────────────────────────────────────────┘
"""
    
    # Agregar artículos si se solicita
    if incluir_articulos:
        resultado += f"""
┌───────────────────────────────────────────────────────────────────────┐
│ BASE CONSTITUCIONAL                                                   │
├───────────────────────────────────────────────────────────────────────┤
│ {l1['nombre']:<20} → {l1['articulo']:<35} │
│ {l2['nombre']:<20} → {l2['articulo']:<35} │
└───────────────────────────────────────────────────────────────────────┘
"""
    
    # Similitudes
    resultado += f"""
═══════════════════════════════════════════════════════════════════════
SIMILITUDES:
• Ambos son mecanismos constitucionales en Colombia
• Ambos protegen derechos reconocidos por la Constitución
• Ambos son de trámite preferente

DIFERENCIAS PRINCIPALES:
• {l1['nombre']} es para {l1['procede'].lower()}
• {l2['nombre']} es para {l2['procede'].lower()}
• Los tiempos de respuesta son diferentes
═══════════════════════════════════════════════════════════════════════
"""
    
    return resultado


# ============================================================================
# EJEMPLO 2: Schema con Literales (Valores Restringidos)
# ============================================================================

class CalcularInteresInput(BaseModel):
    """Schema para cálculo de intereses moratorios."""
    
    capital: float = Field(
        description="Capital base para el cálculo (en pesos colombianos)",
        gt=0  # Mayor que 0
    )
    dias_mora: int = Field(
        description="Número de días de mora",
        ge=0  # Mayor o igual a 0
    )
    tasa_tipo: Literal["corriente", "moratorio"] = Field(
        description="Tipo de tasa a aplicar"
    )


@tool(args_schema=CalcularInteresInput)
def calcular_interes_moratorio(
    capital: float,
    dias_mora: int,
    tasa_tipo: str
) -> str:
    """
    Calcula intereses moratorios según la tasa certificada por la Superintendencia.
    
    Args:
        capital: Capital base para el cálculo
        dias_mora: Número de días de mora
        tasa_tipo: Tipo de tasa (corriente o moratorio)
    
    Returns:
        Detalle del cálculo realizado
    """
    
    # Tasas de ejemplo (en producción, obtener de API de Superintendencia)
    tasas = {
        "corriente": 0.0085,  # 0.85% mensual
        "moratorio": 0.0150   # 1.50% mensual (máximo permitido)
    }
    
    tasa_mensual = tasas[tasa_tipo]
    tasa_diaria = tasa_mensual / 30
    
    # Fórmula: Interés = Capital × Tasa Diaria × Días de Mora
    interes = capital * tasa_diaria * dias_mora
    total = capital + interes
    
    return f"""
╔═══════════════════════════════════════════════════════════════════════╗
                    CÁLCULO DE INTERESES MORATORIOS
╚═══════════════════════════════════════════════════════════════════════╝

DATOS DEL CÁLCULO:
┌───────────────────────────────────────────────────────────────────────┐
│ Capital Base:              ${capital:,.0f} COP                          │
│ Días de Mora:              {dias_mora} días                            │
│ Tipo de Tasa:              {tasa_tipo.capitalize():<20}           │
│ Tasa Mensual:              {tasa_mensual*100:.2f}%                       │
│ Tasa Diaria:               {tasa_diaria*100:.4f}%                       │
└───────────────────────────────────────────────────────────────────────┘

RESULTADOS:
┌───────────────────────────────────────────────────────────────────────┐
│ Interés Calculado:         ${interes:,.0f} COP                          │
│ Total a Pagar:             ${total:,.0f} COP                            │
└───────────────────────────────────────────────────────────────────────┘

FÓRMULA APLICADA:
  Interés = Capital × Tasa Diaria × Días de Mora
  Interés = ${capital:,.0f} × {tasa_diaria*100:.4f}% × {dias_mora}
  Interés = ${interes:,.0f}

═══════════════════════════════════════════════════════════════════════
Nota: Este cálculo es una aproximación. Para casos reales, consulte
      la tasa certificada vigente por la Superintendencia Financiera.
═══════════════════════════════════════════════════════════════════════
"""


# ============================================================================
# EJEMPLO 3: Schema con Listas
# ============================================================================

class BuscarMultipleInput(BaseModel):
    """Schema para búsqueda múltiple de leyes."""
    
    leyes: List[str] = Field(
        description="Lista de nombres de leyes a buscar"
    )
    formato_salida: Literal["resumido", "detallado", "tabla"] = Field(
        default="resumido",
        description="Formato de presentación de resultados"
    )


@tool(args_schema=BuscarMultipleInput)
def buscar_leyes_multiples(
    leyes: List[str],
    formato_salida: str = "resumido"
) -> str:
    """
    Busca múltiples leyes simultáneamente y presenta resultados en el formato solicitado.
    
    Args:
        leyes: Lista de nombres de leyes a buscar
        formato_salida: Formato de presentación (resumido, detallado, tabla)
    
    Returns:
        Resultados de la búsqueda en el formato solicitado
    """
    
    base_datos = {
        "tutela": {"nombre": "Tutela", "articulo": "Art. 86", "tiempo": "10 días"},
        "derecho_peticion": {"nombre": "Derecho de Petición", "articulo": "Art. 23", "tiempo": "15 días"},
        "habeas_corpus": {"nombre": "Habeas Corpus", "articulo": "Art. 30", "tiempo": "36 horas"},
        "accion_popular": {"nombre": "Acción Popular", "articulo": "Art. 88", "tiempo": "Variable"},
        "accion_cumplimiento": {"nombre": "Acción de Cumplimiento", "articulo": "Art. 87", "tiempo": "Según caso"}
    }
    
    resultados = []
    no_encontradas = []
    
    for ley in leyes:
        ley_key = ley.lower().replace(" ", "_")
        if ley_key in base_datos:
            resultados.append(base_datos[ley_key])
        else:
            no_encontradas.append(ley)
    
    # Construir salida según formato
    if formato_salida == "resumido":
        salida = "RESULTADOS DE BÚSQUEDA (Resumido)\n"
        salida += "=" * 60 + "\n\n"
        
        for r in resultados:
            salida += f"✓ {r['nombre']}\n"
            salida += f"  Base: {r['articulo']} | Tiempo: {r['tiempo']}\n\n"
        
        if no_encontradas:
            salida += f"\n❌ No encontradas: {', '.join(no_encontradas)}"
    
    elif formato_salida == "detallado":
        salida = "RESULTADOS DE BÚSQUEDA (Detallado)\n"
        salida += "=" * 60 + "\n\n"
        
        for r in resultados:
            salida += f"┌────────────────────────────────────────────────────┐\n"
            salida += f"│ {r['nombre']:<50} │\n"
            salida += f"├────────────────────────────────────────────────────┤\n"
            salida += f"│ Base Legal: {r['articulo']:<36} │\n"
            salida += f"│ Tiempo:     {r['tiempo']:<36} │\n"
            salida += f"└────────────────────────────────────────────────────┘\n\n"
        
        if no_encontradas:
            salida += f"\n❌ No encontradas: {', '.join(no_encontradas)}"
    
    elif formato_salida == "tabla":
        salida = "RESULTADOS DE BÚSQUEDA (Tabla)\n"
        salida += "=" * 60 + "\n\n"
        salida += f"{'LEY':<25} {'BASE LEGAL':<15} {'TIEMPO':<15}\n"
        salida += "-" * 60 + "\n"
        
        for r in resultados:
            salida += f"{r['nombre']:<25} {r['articulo']:<15} {r['tiempo']:<15}\n"
        
        if no_encontradas:
            salida += f"\n❌ No encontradas: {', '.join(no_encontradas)}"
    
    return salida


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def main():
    """Función principal para demostrar schemas personalizados"""
    
    print("=" * 80)
    print("MÓDULO 5.2 - HERRAMIENTAS CON SCHEMA PERSONALIZADO")
    print("=" * 80)
    
    # 1. Mostrar schemas
    print("\n1️⃣ SCHEMAS DEFINIDOS")
    print("-" * 80)
    
    print(f"\nCompararLeyesInput:")
    print(f"  {CompararLeyesInput.model_json_schema()}")
    
    print(f"\nCalcularInteresInput:")
    print(f"  {CalcularInteresInput.model_json_schema()}")
    
    # 2. Probar herramientas directamente
    print("\n2️⃣ PRUEBA DIRECTA DE HERRAMIENTAS")
    print("-" * 80)
    
    # Comparar leyes
    print("\n📊 Comparación de Leyes:")
    resultado = comparar_leyes.invoke({
        "ley1": "tutela",
        "ley2": "habeas_corpus",
        "incluir_tiempos": True,
        "incluir_articulos": True
    })
    print(resultado)
    
    # Calcular interés
    print("\n💰 Cálculo de Intereses:")
    resultado = calcular_interes_moratorio.invoke({
        "capital": 1000000,
        "dias_mora": 30,
        "tasa_tipo": "moratorio"
    })
    print(resultado)
    
    # Búsqueda múltiple
    print("\n🔍 Búsqueda Múltiple:")
    resultado = buscar_leyes_multiples.invoke({
        "leyes": ["tutela", "habeas_corpus", "accion_popular", "ley_ficticia"],
        "formato_salida": "tabla"
    })
    print(resultado)
    
    # 3. Usar con agente
    print("\n3️⃣ USAR HERRAMIENTAS CON AGENTE")
    print("-" * 80)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
    )
    
    agent = create_agent(
        llm,
        tools=[comparar_leyes, calcular_interes_moratorio, buscar_leyes_multiples],
        system_prompt="""Eres un asistente legal experto en derecho colombiano.
        
        Tus capacidades:
        - Comparar mecanismos constitucionales
        - Calcular intereses moratorios
        - Buscar múltiples leyes simultáneamente
        
        Instrucciones:
        - Usa las herramientas para obtener información precisa
        - Presenta los resultados de manera clara
        - Responde en español
        """
    )
    
    # Probar con pregunta compleja
    pregunta = "Necesito comparar la tutela con el habeas corpus, incluyendo tiempos y artículos"
    print(f"\nPregunta: {pregunta}")
    print("-" * 80)
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": pregunta}]
    })
    
    print(f"\nRespuesta del Agente:\n{response['messages'][-1].content[:500]}...")


# ============================================================================
# MEJORES PRÁCTICAS
# ============================================================================

"""
✅ MEJORES PRÁCTICAS PARA SCHEMAS:

1. Usa Field con descripciones claras
   campo: str = Field(description="Descripción detallada")

2. Valida con restricciones
   edad: int = Field(gt=0, lt=150)  # Mayor que 0, menor que 150

3. Usa Literal para valores fijos
   tipo: Literal["opcion1", "opcion2"]

4. Valores por defecto claros
   incluir_detalle: bool = Field(default=True)

5. Nombres descriptivos
   class CalcularInteresInput(BaseModel)  # ✅
   class Input(BaseModel)  # ❌
"""


if __name__ == "__main__":
    main()
