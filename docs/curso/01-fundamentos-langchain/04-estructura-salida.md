# Módulo 1.4: Estructura de Salida

## Objetivos
- Comprender los Output Parsers de LangChain
- Usar PydanticOutputParser para structured output
- Extraer datos estructurados del LLM
- Validar respuestas del modelo

---

## 1.4.1 ¿Por qué Structured Output?

Los LLMs por defecto retornan texto libre. Para aplicaciones reales necesitamos:

| Caso de Uso | Sin Estructura | Con Estructura |
|-------------|----------------|----------------|
| **Extraer fechas** | "El 15 de marzo de 2024" | `{"fecha": "2024-03-15"}` |
| **Clasificar** | "Esto parece derecho penal" | `{"categoria": "penal", "confianza": 0.95}` |
| **Múltiples items** | Texto con varias leyes | `{"leyes": [{"nombre": "...", "articulo": "..."}]}` |

---

## 1.4.2 Código de Ejemplo

Archivo: `src/course_examples/modulo_01/04_structured_output.py`

```python
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
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

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
    print(f"Tipo: {type(resultado)}")  # Ahora es una lista Python
    
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
    
    # Definir el formato esperado
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
    print(f"Tipo: {type(resultado)}")  # Dict Python
    print(f"Ley: {resultado.get('ley', 'N/A')}")


# ============================================================
# 3. PYDANTIC OUTPUT PARSER (RECOMENDADO)
# ============================================================

# Definir schema con Pydantic
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
    
    from langchain_core.output_parsers import PydanticOutputParser
    
    # Crear parser desde el schema
    parser = PydanticOutputParser(pydantic_object=LeyInfo)
    
    # Crear prompt con instrucciones de formato
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extrae información legal del texto."),
        ("user", """Texto: {texto}
        
{format_instructions}

Extrae la información de la ley mencionada.""")
    ])
    
    # Insertar instrucciones de formato en el prompt
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    # Crear chain
    chain = prompt | llm | parser
    
    texto = """
    La Ley 10 de 1991, conocida como la Ley de Carrera Administrativa,
    establece el sistema de empleo público en Colombia.
    """
    
    resultado = chain.invoke({"texto": texto})
    
    print(f"Resultado: {resultado}")
    print(f"Tipo: {type(resultado)}")  # Instancia de LeyInfo
    print(f"Nombre: {resultado.nombre}")
    print(f"Año: {resultado.ano}")
    print(f"Descripción: {resultado.descripcion}")


# ============================================================
# 4. LISTA DE OBJETOS CON PYDANTIC
# ============================================================

class ArticuloLegal(BaseModel):
    """Artículo de una ley"""
    numero: str = Field(description="Número del artículo")
    contenido: str = Field(description="Contenido/resumen del artículo")
    tipo: str = Field(description="Tipo: normativo, procedimental, sancionatorio")


class LeyCompleta(BaseModel):
    """Ley completa con múltiples artículos"""
    nombre: str = Field(description="Nombre de la ley")
    ano: int = Field(description="Año de la ley")
    pais: str = Field(description="País")
    articulos: List[ArticuloLegal] = Field(description="Lista de artículos")


def lista_de_objetos():
    """Extraer lista de objetos estructurados"""
    
    print("\n" + "=" * 60)
    print("LISTA DE OBJETOS CON PYDANTIC")
    print("=" * 60)
    
    from langchain_core.output_parsers import PydanticOutputParser
    
    parser = PydanticOutputParser(pydantic_object=LeyCompleta)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza el documento legal y extrae información estructurada."),
        ("user", """Documento:
        {documento}
        
        {format_instructions}
        
        Extrae el nombre, año y los artículos mencionados.""")
    ])
    
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    
    documento = """
    LEY 50 DE 1990
    
    Artículo 1: Objeto. La presente ley tiene por objeto promover el empleo.
    
    Artículo 2: Contrato a término fijo. El contrato de trabajo a término 
    fijo no puede exceder de tres años.
    
    Artículo 3: Periodo de prueba. El periodo de prueba no puede exceder 
    de dos meses.
    """
    
    resultado = chain.invoke({"documento": documento})
    
    print(f"Ley: {resultado.nombre} de {resultado.ano}")
    print(f"País: {resultado.pais}")
    print(f"Artículos: {len(resultado.articulos)}")
    
    for articulo in resultado.articulos:
        print(f"  - Art. {articulo.numero}: {articulo.tipo}")


# ============================================================
# 5. WITH_STRUCTURED_OUTPUT (GEMINI NATIVO)
# ============================================================

def with_structured_output():
    """Usar el método nativo de Gemini para structured output"""
    
    print("\n" + "=" * 60)
    print("WITH_STRUCTURED_OUTPUT (NATIVO)")
    print("=" * 60)
    
    # Definir schema
    class Persona(BaseModel):
        nombre: str
        edad: int
        profesion: str
        habilidades: List[str]
    
    # Configurar el modelo con structured output
    llm_structured = llm.with_structured_output(Persona)
    
    # Usar directamente
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Crea una persona ficticia que sea abogado en Bogotá")
    ])
    
    chain = prompt | llm_structured
    
    resultado = chain.invoke({})
    
    print(f"Persona creada: {resultado}")
    print(f"Tipo: {type(resultado)}")  # Instancia de Persona
    print(f"Nombre: {resultado.nombre}")
    print(f"Profesión: {resultado.profesion}")
    print(f"Habilidades: {resultado.habilidades}")


# ============================================================
# 6. VALIDACIÓN Y MANEJO DE ERRORES
# ============================================================

def validacion_y_errores():
    """Manejar errores de parsing"""
    
    print("\n" + "=" * 60)
    print("VALIDACIÓN Y MANEJO DE ERRORES")
    print("=" * 60)
    
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import ValidationError
    
    class RespuestaLegal(BaseModel):
        pregunta: str
        respuesta: str
        confianza: float = Field(ge=0.0, le=1.0)  # Entre 0 y 1
        fuentes: List[str]
    
    parser = PydanticOutputParser(pydantic_object=RespuestaLegal)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responde preguntas legales con precisión."),
        ("user", """Pregunta: {pregunta}
        
        {format_instructions}
        
        Responde con honestidad. Si no sabes, indica baja confianza.""")
    ])
    
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    
    try:
        resultado = chain.invoke({
            "pregunta": "¿Qué es una acción de grupo?"
        })
        
        print(f"✅ Parsing exitoso")
        print(f"Pregunta: {resultado.pregunta}")
        print(f"Confianza: {resultado.confianza}")
        print(f"Fuentes: {resultado.fuentes}")
        
    except ValidationError as e:
        print(f"❌ Error de validación: {e}")
        print("   El LLM no retornó el formato esperado")
        
    except Exception as e:
        print(f"❌ Error inesperado: {type(e).__name__}: {e}")


# ============================================================
# 7. CASO DE USO: EXTRACCIÓN LEGAL
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
    lista_de_objetos()
    with_structured_output()
    validacion_y_errores()
    extractor_legal()

```

---

## 1.4.3 Métodos de Structured Output

### 1. StrOutputParser
El más básico - solo retorna el string:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
resultado = chain.invoke({})  # str
```

### 2. JsonOutputParser
Retorna un dict Python:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | llm | parser
resultado = chain.invoke({})  # dict
```

### 3. PydanticOutputParser (Recomendado)
Retorna un objeto Pydantic validado:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class MiSchema(BaseModel):
    campo1: str = Field(description="...")
    campo2: int

parser = PydanticOutputParser(pydantic_object=MiSchema)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())
chain = prompt | llm | parser
resultado = chain.invoke({})  # Instancia de MiSchema
```

### 4. with_structured_output (Nativo Gemini)
El más simple si usas Gemini:

```python
llm_structured = llm.with_structured_output(MiSchema)
chain = prompt | llm_structured
resultado = chain.invoke({})  # Instancia de MiSchema
```

---

## 1.4.4 Mejores Prácticas

### ✅ DO

```python
# 1. Usa temperature baja para structured output
llm = ChatGoogleGenerativeAI(temperature=0.3)

# 2. Define schemas claros con descripciones
class MiSchema(BaseModel):
    campo: str = Field(description="Descripción clara del campo")

# 3. Usa with_structured_output cuando sea posible (más simple)
llm_structured = llm.with_structured_output(MiSchema)

# 4. Maneja errores de validación
try:
    resultado = chain.invoke({})
except ValidationError as e:
    logger.error(f"LLM no siguió el formato: {e}")
```

### ❌ DON'T

```python
# 1. No uses temperature alta
llm = ChatGoogleGenerativeAI(temperature=1.5)  # ❌ Para structured

# 2. No uses schemas sin descripciones
class MiSchema(BaseModel):
    campo1: str  # ❌ Sin Field description

# 3. No asumas que siempre funcionará
resultado = chain.invoke({})  # ❌ Sin try/except
```

---

## 1.4.5 Ejercicios Prácticos

### Ejercicio 1: Extractor de Sentencias

Crea un schema Pydantic para extraer:
- Número de sentencia
- Fecha
- Corte/Tribunal
- Partes involucradas
- Decisión

### Ejercicio 2: Clasificador de Documentos

Crea un parser que clasifique documentos como:
- `{"tipo": "Ley"|"Decreto"|"Sentencia"|"Contrato", "confianza": 0.0-1.0}`

### Ejercicio 3: Extractor de Fechas

Extrae todas las fechas de un documento:
- `{"fechas": [{"fecha": "YYYY-MM-DD", "evento": "descripción"}]}`

---

## 1.4.6 Recursos Adicionales

### Documentación Oficial
- [LangChain Output Parsers](https://docs.langchain.com/oss/python/langchain/concepts/output_parsers)
- [PydanticOutputParser](https://docs.langchain.com/oss/python/langchain/integrations/output_parsers/pydantic)
- [Gemini Structured Output](https://docs.langchain.com/oss/python/langchain/integrations/google_genai/chat#structured-output)

### Siguiente Módulo
➡️ **Módulo 2: Memoria y Contexto**

---

*Lección creada: 2026-03-29*
