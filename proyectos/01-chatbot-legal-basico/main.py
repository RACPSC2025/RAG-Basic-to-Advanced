"""
Chatbot Legal Básico - Punto de Entrada

Este es el punto de entrada principal para ejecutar el Chatbot Legal.

Uso:
    python main.py

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
"""

import sys
import logging
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatbot import LegalChatbot, get_default_chatbot
from src.config import APP_NAME, APP_VERSION, LOG_LEVEL

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def mostrar_bienvenida():
    """Mostrar mensaje de bienvenida."""
    print("\n" + "=" * 80)
    print(f"⚖️  {APP_NAME} v{APP_VERSION}")
    print("=" * 80)
    print("\n¡Hola! Soy tu asistente legal virtual.")
    print("Puedo ayudarte con consultas básicas sobre derecho colombiano.")
    print("\nComandos disponibles:")
    print("  [historial] - Ver historial de conversación")
    print("  [stats]     - Ver estadísticas")
    print("  [reset]     - Resetear conversación")
    print("  [salir]     - Salir del chatbot")
    print("=" * 80)


def main():
    """Función principal del chatbot."""
    
    logger.info(f"Iniciando {APP_NAME} v{APP_VERSION}")
    
    # Crear chatbot
    chatbot = get_default_chatbot()
    
    # Mostrar bienvenida
    mostrar_bienvenida()
    
    # Bucle principal de conversación
    while True:
        try:
            # Obtener input del usuario
            user_input = input("\n👤 Tú: ").strip()
            
            # Verificar comandos especiales
            if user_input.lower() in ["salir", "exit", "quit", "q"]:
                print("\n¡Gracias por usar el Chatbot Legal! ¡Hasta pronto! 👋")
                logger.info("Usuario salió del chatbot")
                break
            
            elif user_input.lower() == "historial":
                history = chatbot.get_conversation_history()
                print(f"\n📜 Historial ({len(history)} mensajes):")
                for i, msg in enumerate(history[-10:], 1):  # Últimos 10
                    emoji = "👤" if msg["role"] == "user" else "🤖"
                    print(f"  {i}. {emoji} {msg['content'][:100]}...")
                continue
            
            elif user_input.lower() == "stats":
                stats = chatbot.get_stats()
                print(f"\n📊 Estadísticas:")
                print(f"  Turnos: {stats['turn_count']}")
                print(f"  Mensajes en memoria: {stats['short_term_messages']}")
                print(f"  HITL habilitado: {stats['hitl_enabled']}")
                continue
            
            elif user_input.lower() == "reset":
                chatbot.reset_conversation()
                print("\n✅ Conversación reseteada")
                logger.info("Conversación reseteada por usuario")
                continue
            
            # Verificar input vacío
            if not user_input:
                print("⚠️  Por favor ingresa una consulta")
                continue
            
            # Obtener respuesta del chatbot
            print("\n🤖 Escribiendo...", end="\r")
            response = chatbot.chat(user_input, auto_approve=False)
            
            # Mostrar respuesta
            print(f"\n🤖 Asistente: {response['respuesta']}")
            
            # Mostrar indicadores adicionales
            if response.get("edited_by_human"):
                print("  ✏️  Respuesta editada por humano")
            elif response.get("rejected_by_human"):
                print("  ⚠️  Respuesta rechazada por humano")
            
            if response.get("confidence"):
                print(f"  📊 Confianza: {response['confidence']:.0%}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Chatbot interrumpido por el usuario")
            logger.info("Chatbot interrumpido por Ctrl+C")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Error en main loop: {e}", exc_info=True)
    
    logger.info(f"{APP_NAME} finalizado")


if __name__ == "__main__":
    main()
