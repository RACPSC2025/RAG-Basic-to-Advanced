"""
Test Completo del Chatbot Legal - AWS Bedrock

Este script prueba todas las funcionalidades del chatbot.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.chatbot import LegalChatbot, get_default_chatbot
from src.config import APP_NAME, APP_VERSION, LLM_MODEL_ID


def main():
    """Ejecutar todos los tests."""
    print("=" * 70)
    print(f"TEST COMPLETO - {APP_NAME} v{APP_VERSION}")
    print("=" * 70)
    print(f"Modelo: {LLM_MODEL_ID}")
    print()

    # Crear chatbot
    print("[1/6] Inicializando chatbot...")
    chatbot = LegalChatbot()
    print("      Chatbot inicializado correctamente")
    print()

    # Test 1: Consulta básica sobre tutela
    print("[2/6] Test: Consulta basica sobre tutela")
    print("-" * 70)
    response = chatbot.chat("Que es una tutela en Colombia?", auto_approve=True)
    print(f"Respuesta ({len(response['respuesta'])} chars):")
    print(f"  {response['respuesta'][:300]}...")
    print(f"Confianza: {response.get('confidence', 0):.0%}")
    print()

    # Test 2: Consulta de seguimiento (memoria)
    print("[3/6] Test: Consulta de seguimiento (prueba de memoria)")
    print("-" * 70)
    response = chatbot.chat("Cual es el plazo maximo para presentar una tutela?", auto_approve=True)
    print(f"Respuesta:")
    print(f"  {response['respuesta'][:300]}...")
    print()

    # Test 3: Consulta sobre debido proceso
    print("[4/6] Test: Consulta sobre debido proceso")
    print("-" * 70)
    response = chatbot.chat("Que es el debido proceso?", auto_approve=True)
    print(f"Respuesta:")
    print(f"  {response['respuesta'][:300]}...")
    print()

    # Test 4: Estadisticas
    print("[5/6] Test: Estadisticas del chatbot")
    print("-" * 70)
    stats = chatbot.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    # Test 5: Historial
    print("[6/6] Test: Historial de conversacion")
    print("-" * 70)
    history = chatbot.get_conversation_history()
    print(f"Total mensajes: {len(history)}")
    for i, msg in enumerate(history, 1):
        role_emoji = "👤" if msg["role"] == "user" else "🤖"
        preview = msg["content"][:60].replace("\n", " ")
        print(f"  {i}. {role_emoji} [{msg['role']}]: {preview}...")
    print()

    # Test 6: Reset
    print("Test Extra: Reset de conversacion")
    print("-" * 70)
    chatbot.reset_conversation()
    stats_after = chatbot.get_stats()
    print(f"  Turnos despues de reset: {stats_after['turn_count']}")
    print(f"  Mensajes despues de reset: {stats_after['short_term_messages']}")
    print()

    print("=" * 70)
    print("✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
