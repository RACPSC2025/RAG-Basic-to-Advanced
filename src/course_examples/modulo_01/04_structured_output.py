"""
04_structured_output.py
Estructura de Salida con LangChain

Objetivo: Dominar Output Parsers y structured output
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    NumberedListOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,  # Más determinista para structured output
)


# ============================================================
# 1. OUTPUT PARSERS BÁSICOS
# ============================================================

def output_parsers_basicos():
    """Usar output parsers básicos"""
    
    print("=" * 60)
    print("OUTPUT PARSERS BÁSICOS")
    print("=" * 60)
    
    # 1. StrOutputParser (por defecto)
    print("\n--- StrOutputParser ---")
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Di hola en 3 idiomas")
    ])
    
    chain = prompt | llm | StrOutputParser()
    resultado = chain.invoke({})
    print(f"String: {resultado}")
    
    # 2. CommaSeparatedListOutputParser
    print("\n--- CommaSeparatedListOutputParser ---")
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Nombra 5 países de Sudamérica, separados por comas")
    ])
    
    parser = CommaSeparatedListOutputParser()
    chain = prompt | llm | parser
    resultado = chain.invoke({})
    print(f"Lista: {resultado}")
    print(f"Tipo: {type(resultado)}")
    
    # 3. NumberedListOutputParser
    print("\n--- NumberedListOutputParser ---")
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Nombra 3 derechos fundamentales en formato de lista numerada")
    ])
    
    parser = NumberedListOutputParser()
    chain = prompt | llm | parser
    resultado = chain.invoke({})
    print(f"Lista numerada: {resultado}")


# ============================================================
# 2. JSON OUTPUT PARSER
# ============================================================

def json_output_parser():
    """Extraer JSON estructurado"""
    
    print("\n" + "=" * 60)
    print("JSON OUTPUT PARSER")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extrae la información del texto en formato JSON"),
        ("user", """Texto: {texto}
        
Extrae:
- Nombre de la ley
- Número de la ley
- Año
- Descripción breve

Responde SOLO con JSON válido.""")
    ])
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    texto = """
    La Ley 1564 de 2012 es el Código General del Proceso en Colombia.
    Establece las reglas de procedimiento para procesos judiciales.
    """
    
    resultado = chain.invoke({"texto": texto})
    
    print(f"JSON extraído: {resultado}")
    print(f"Tipo: {type(resultado)}")
    print(f"Ley: {resultado.get('ley', 'N/A')}")


# ============================================================
# 3. PYDANTIC OUTPUT PARSER (RECOMENDADO)
# ============================================================

class LeyInfo(BaseModel):
    """Información sobre una ley"""
    nombre: str = Field(description="Nombre completo de la ley")
    numero: str = Field(description="Número de la ley")
    ano: int = Field(description="Año de la ley")
    descripcion: str = Field(description="Descripción breve en una frase")
    pais: str = Field(default="Colombia", description="País de origen")


def pydantic_output_parser():
    """Usar Pydantic para structured output"""
    
    print("\n" + "=" * 60)
    print("PYDANTIC OUTPUT PARSER")
    print("=" * 60)
    
    parser = PydanticOutputParser(pydantic_object=LeyInfo)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extrae información legal del texto."),
        ("user", """Texto: {texto}
        
{format_instructions}

Extrae la información de la ley mencionada.""")
    ])
    
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    
    texto = """
    La Ley 10 de 1991, conocida como la Ley de Carrera Administrativa,
    establece el sistema de empleo público en Colombia.
    """
    
    resultado = chain.invoke({"texto": texto})
    
    print(f"Resultado: {resultado}")
    print(f"Tipo: {type(resultado)}")
    print(f"Nombre: {resultado.nombre}")
    print(f"Año: {resultado.ano}")


# ============================================================
# 4. WITH_STRUCTURED_OUTPUT (GEMINI NATIVO)
# ============================================================

def with_structured_output():
    """Usar el método nativo de Gemini para structured output"""
    
    print("\n" + "=" * 60)
    print("WITH_STRUCTURED_OUTPUT (NATIVO)")
    print("=" * 60)
    
    class Persona(BaseModel):
        nombre: str
        edad: int
        profesion: str
        habilidades: List[str]
    
    llm_structured = llm.with_structured_output(Persona)
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Crea una persona ficticia que sea abogado en Bogotá")
    ])
    
    chain = prompt | llm_structured
    
    resultado = chain.invoke({})
    
    print(f"Persona creada: {resultado}")
    print(f"Tipo: {type(resultado)}")
    print(f"Nombre: {resultado.nombre}")
    print(f"Profesión: {resultado.profesion}")


# ============================================================
# 5. CASO DE USO: EXTRACCIÓN LEGAL
# ============================================================

class DocumentoLegal(BaseModel):
    """Documento legal estructurado"""
    tipo: str = Field(description="Tipo: Ley, Decreto, Sentencia, Contrato")
    numero: str = Field(description="Número de identificación")
    fecha: str = Field(description="Fecha en formato YYYY-MM-DD")
    entidad_emisora: str = Field(description="Entidad que emitió el documento")
    resumen: str = Field(description="Resumen en 2-3 frases")
    palabras_clave: List[str] = Field(description="5 palabras clave")
    relevancia: float = Field(ge=0.0, le=1.0, description="Relevancia para derecho laboral")


def extractor_legal():
    """Extractor de documentos legales"""
    
    print("\n" + "=" * 60)
    print("CASO DE USO: EXTRACCIÓN LEGAL")
    print("=" * 60)
    
    llm_structured = llm.with_structured_output(DocumentoLegal)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza documentos legales y extrae información estructurada."),
        ("user", """Documento:
        {documento}
        
        Extrae toda la información relevante.""")
    ])
    
    chain = prompt | llm_structured
    
    documento = """
    CORTE CONSTITUCIONAL
    SENTENCIA T-123 DE 2024
    
    Magistrado Ponente: Dr. Juan Pérez
    
    Acción de Tutela presentada por María Rodríguez contra EPS Sanitas.
    
    El paciente requiere cirugía urgente y la EPS niega el procedimiento.
    
    Fecha: 15 de marzo de 2024
    """
    
    resultado = chain.invoke({"documento": documento})
    
    print(f"Tipo: {resultado.tipo}")
    print(f"Número: {resultado.numero}")
    print(f"Fecha: {resultado.fecha}")
    print(f"Entidad: {resultado.entidad_emisora}")
    print(f"Resumen: {resultado.resumen[:100]}...")
    print(f"Palabras clave: {resultado.palabras_clave}")
    print(f"Relevancia laboral: {resultado.relevancia}")


if __name__ == "__main__":
    output_parsers_basicos()
    json_output_parser()
    pydantic_output_parser()
    with_structured_output()
    extractor_legal()
