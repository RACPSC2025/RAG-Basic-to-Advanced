import os
from typing import List, Optional
from llama_index.core import Document
from llama_index.readers.file import PDFReader
import PyPDF2
import fitz  # PyMuPDF
import pytesseract
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.preprocessor import ImagePreprocessor

from dotenv import load_dotenv


load_dotenv()