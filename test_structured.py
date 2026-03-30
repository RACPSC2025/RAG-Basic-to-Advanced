"""
Test structured output.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

from proyectos.Rag_Legal.config import get_llm
from pydantic import BaseModel, Field

class GradeOutput(BaseModel):
    score: str = Field(description="'si' o 'no'")
    razon: str = Field(description="Explicacion")

print("Probando structured output...")
try:
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeOutput)
    print("✅ structured_llm inicializado.")
    
    result = structured_llm.invoke("Responde 'si' explicacion 'test'.")
    print(f"Resultado: {result}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
