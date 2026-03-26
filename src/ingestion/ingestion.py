import os
from dotenv import load_dotenv
import cv2
from PIL import Image

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_parse import LlamaParse

# from llama_index.core.node_parser import SimpleNodeParser, SentenceWindowNodeParser

# from llama_index.llms.google_genai import google_genai
# from numpy import result_type

load_dotenv()

# Configuracion de LLM
# Settings.llm = google_genai(
#     model= os.getenv("GOOGLE_GEMINI_MODEL"),
#     temperature=os.getenv("GOOGLE_GEMINI_TEMPERATURE"),
#     max_tokens=os.getenv("GOOGLE_GEMINI_MAX_TOKENS")
# )

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

def process_images(directory_path):
    # Lee las imágenes desde el directorio
    images = SimpleDirectoryReader(directory_path).load_data()
    
    # Procesa cada imagen con LlamaParse
    processed_images = []
    for img in images:
        processed_content = llama_parser.parse(img.get_content())
        processed_images.append(processed_content)
    
    return processed_images

def process_documents(directory_path):
    # Lee los documentos desde el directorio
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    # Procesa cada documento con LlamaParse
    processed_documents = []
    for doc in documents:
        processed_content = llama_parser.parse(doc.get_content())
        processed_documents.append(processed_content)
    
    return processed_documents