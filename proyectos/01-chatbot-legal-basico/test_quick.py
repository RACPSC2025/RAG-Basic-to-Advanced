"""
Test rápido del Chatbot Legal - AWS Bedrock

Verifica que el chatbot pueda inicializarse y responder.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatbot import LegalChatbot, get_default_chatbot


def test_imports():
    """Verificar que todas las importaciones funcionen."""
    print("✓ Importaciones exitosas")
    return True


def test_chatbot_creation():
    """Verificar que el chatbot se pueda crear."""
    chatbot = LegalChatbot()
    print("✓ Chatbot creado exitosamente")
    print(f"  - App: {chatbot.memory}")
    print(f"  - Turnos: {chatbot.memory.turn_count}")
    return chatbot


def test_chat_response(chatbot):
    """Verificar que el chatbot pueda responder."""
    print("\nProbando consulta básica...")
    response = chatbot.chat("¿Qué es una tutela?", auto_approve=True)
    
    print(f"✓ Respuesta recibida ({len(response['respuesta'])} chars)")
    print(f"  - Confianza: {response.get('confidence', 0):.0%}")
    print(f"  - Respuesta: {response['respuesta'][:200]}...")
    return True


def test_stats(chatbot):
    """Verificar estadísticas del chatbot."""
    stats = chatbot.get_stats()
    print("\nEstadísticas del chatbot:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    return True


def main():
    """Ejecutar tests básicos."""
    print("=" * 60)
    print("TEST RÁPIDO - CHATBOT LEGAL (AWS BEDROCK)")
    print("=" * 60)
    
    try:
        # Test 1: Importaciones
        test_imports()
        
        # Test 2: Creación del chatbot
        chatbot = test_chatbot_creation()
        
        # Test 3: Respuesta
        test_chat_response(chatbot)
        
        # Test 4: Estadísticas
        test_stats(chatbot)
        
        print("\n" + "=" * 60)
        print("✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
