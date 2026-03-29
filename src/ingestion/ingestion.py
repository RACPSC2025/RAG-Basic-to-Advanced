import os
from dotenv import load_dotenv
import cv2


# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_parse import LlamaParse


load_dotenv()

# Parseador de tablas en español con formato Markdown
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")

if LLAMA_PARSE_API_KEY is None:
    raise ValueError("LLAMA_PARSE_API_KEY no está configurada en las variables de entorno.")
else:
    llama_parser = LlamaParse(
        api_key=LLAMA_PARSE_API_KEY,
        result_type="markdown",
        language="es",
        verbose=True
)

def process_documents(directory_path):
    # Lee los documentos desde el directorio
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    # Procesa cada documento con LlamaParse
    processed_documents = []
    for doc in documents:
        processed_content = llama_parser.parse(doc.get_content())
        processed_documents.append(processed_content)
    
    return processed_documents