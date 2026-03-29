import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()

def setup_logger(name: str = "LegalRAG") -> logging.Logger:
    """Configurar logger con archivo y consola"""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Crear directorio de logs si no existe
    os.environ["LOGS_PATH"].mkdir(parents=True, exist_ok=True)
    
    # Nombre del archivo con fecha
    log_file = os.environ["LOGS_PATH"] / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Evitar duplicados
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()