# ⚖️ Auditor Legal Inteligente (CRAG + Self-RAG)

Este es el **Proyecto 06** del curso avanzado de LangChain + LangGraph. Implementa un pipeline de auditoría legal que garantiza la precisión mediante la verificación continua de hechos.

## 🚀 Características
- **Corrective RAG (CRAG)**: Valida si los documentos recuperados son realmente útiles antes de generar.
- **Self-RAG**: El modelo se autocrítica para detectar alucinaciones y asegurar que la respuesta resuelva la duda del usuario.
- **ChromaDB por defecto**: Configurado para funcionar localmente.
- **Qdrant Cloud Ready**: Código listo (comentado) para conectar con Qdrant Cloud usando las credenciales del `.env`.

## 🛠️ Requisitos
1. Tener configurado el archivo `.env` en la raíz con:
   - `GOOGLE_API_KEY`
   - `QDRANT_URL` (para uso futuro con Qdrant Cloud)
   - `QDRANT_API_KEY` (para uso futuro con Qdrant Cloud)
2. Instalar dependencias: 
   ```bash
   pip install langgraph langchain-google-genai pydantic langchain-chroma
   ```

## 📂 Estructura
- `main.py`: Ejecución del sistema.
- `src/agent/`: Lógica del grafo de LangGraph.
- `src/config.py`: Configuración de Chroma y Qdrant.
- `docs/PROYECTO_6_DETALLES.md`: Documentación técnica completa.

## 📝 Uso
Para que el sistema funcione, debes tener documentos en tu base de datos Chroma. Si la base de datos está vacía, el sistema activará la **Transformación de Consulta** (CRAG).

Ejecuta el auditor con:
```bash
python proyectos/06-auditor-legal-crag-selfrag/main.py
```

---
*Refactorizado para ChromaDB - 2026-03-29*
