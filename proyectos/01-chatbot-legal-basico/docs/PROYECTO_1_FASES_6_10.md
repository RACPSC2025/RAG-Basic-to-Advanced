# 🤖 Proyecto 1: Chatbot Legal Básico (Fases 6-10)

> **Continuación de la documentación del Proyecto 1**

---

## Fase 6: Memoria Corto Plazo

### Objetivo

Implementar memoria de conversación a corto plazo usando ConversationBufferWindowMemory.

### Paso 6.1: Módulo de Memoria

```python
# src/memory.py
"""
Módulo para gestión de memoria del Chatbot Legal.

Implementa memoria a corto plazo (buffer) y largo plazo (resumen).
"""

import logging
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import MEMORY_SHORT_TERM_K, MEMORY_LONG_TERM_ENABLED, LLM_MODEL
import os

# Configurar logging
logger = logging.getLogger(__name__)


class ChatMemory:
    """
    Gestor de memoria para conversaciones legales.
    
    Combina memoria a corto plazo (últimos K turnos)
    con memoria a largo plazo (resumen de la conversación).
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """
        Inicializar gestor de memoria.
        
        Args:
            llm: Instancia del LLM para resumen de largo plazo
        """
        logger.info("Inicializando gestor de memoria")
        
        # Memoria a corto plazo (buffer con ventana)
        self.short_term_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=MEMORY_SHORT_TERM_K,
            input_key="input",
            output_key="output"
        )
        
        logger.debug(f"Memoria corto plazo: k={MEMORY_SHORT_TERM_K}")
        
        # Memoria a largo plazo (resumen)
        if MEMORY_LONG_TERM_ENABLED:
            if llm is None:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    temperature=0.0,  # Temperature baja para resúmenes precisos
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            
            self.long_term_memory = ConversationSummaryMemory(
                llm=llm,
                memory_key="summary",
                return_messages=False,
                input_key="input",
                output_key="output"
            )
            
            logger.info("Memoria largo plazo habilitada")
        else:
            self.long_term_memory = None
            logger.info("Memoria largo plazo deshabilitada")
        
        # Contador de turnos
        self.turn_count = 0
        
        logger.info("Gestor de memoria inicializado")
    
    def add_message(self, user_input: str, ai_output: str):
        """
        Agregar mensaje a la memoria.
        
        Args:
            user_input: Input del usuario
            ai_output: Output del asistente
        """
        logger.debug(f"Agregando mensaje a memoria (turno {self.turn_count + 1})")
        
        # Agregar a memoria corto plazo
        self.short_term_memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )
        
        # Agregar a memoria largo plazo (resumen)
        if self.long_term_memory:
            self.long_term_memory.save_context(
                {"input": user_input},
                {"output": ai_output}
            )
        
        self.turn_count += 1
        
        logger.debug(f"Mensaje agregado. Turnos totales: {self.turn_count}")
    
    def get_history(self) -> List[BaseMessage]:
        """
        Obtener historial de conversación.
        
        Returns:
            Lista de mensajes (HumanMessage y AIMessage)
        """
        chat_history = self.short_term_memory.load_memory_variables({})["chat_history"]
        
        logger.debug(f"Obteniendo historial: {len(chat_history)} mensajes")
        
        return chat_history
    
    def get_summary(self) -> str:
        """
        Obtener resumen de la conversación (largo plazo).
        
        Returns:
            Resumen de la conversación
        """
        if not self.long_term_memory:
            return ""
        
        summary = self.long_term_memory.load_memory_variables({})["summary"]
        
        logger.debug(f"Obteniendo resumen: {len(summary)} caracteres")
        
        return summary
    
    def get_context(self) -> str:
        """
        Obtener contexto completo (historial + resumen).
        
        Returns:
            Contexto completo de la conversación
        """
        context_parts = []
        
        # Agregar resumen si existe
        summary = self.get_summary()
        if summary:
            context_parts.append(f"Resumen de la conversación:\n{summary}\n")
        
        # Agregar historial reciente
        history = self.get_history()
        if history:
            history_text = "\n".join([
                f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
                for msg in history[-MEMORY_SHORT_TERM_K:]
            ])
            context_parts.append(f"Últimos intercambios:\n{history_text}")
        
        context = "\n\n".join(context_parts)
        
        logger.debug(f"Contexto generado: {len(context)} caracteres")
        
        return context
    
    def clear(self):
        """
        Limpiar toda la memoria.
        """
        logger.info("Limpiando memoria")
        
        self.short_term_memory.clear()
        
        if self.long_term_memory:
            self.long_term_memory.clear()
        
        self.turn_count = 0
        
        logger.info("Memoria limpiada")
    
    def get_stats(self) -> Dict:
        """
        Obtener estadísticas de la memoria.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "turn_count": self.turn_count,
            "short_term_messages": len(self.get_history()),
            "long_term_enabled": self.long_term_memory is not None
        }
        
        if self.long_term_memory:
            stats["summary_length"] = len(self.get_summary())
        
        return stats
```

---

## Fase 7: Memoria Largo Plazo

La memoria a largo plazo ya está implementada en el módulo anterior usando `ConversationSummaryMemory`. Esta memoria:

1. **Resume automáticamente** la conversación
2. **Mantiene el contexto** más allá de los últimos K turnos
3. **Se actualiza** con cada nuevo intercambio

### Uso de Memoria Largo Plazo

```python
# Ejemplo de uso
from src.memory import ChatMemory

memory = ChatMemory()

# Después de varias interacciones
summary = memory.get_summary()
print(f"Resumen: {summary}")

# El resumen se usa automáticamente en get_context()
context = memory.get_context()
```

---

## Fase 8: Human in the Loop

### Objetivo

Implementar aprobación humana para respuestas críticas o de baja confianza.

### Paso 8.1: Módulo de Human in the Loop

```python
# src/human_in_loop.py
"""
Módulo para Human in the Loop (HITL).

Permite aprobación humana para respuestas críticas.
"""

import logging
from typing import Optional, Tuple, Literal
from .config import HITL_ENABLED, HITL_CONFIDENCE_THRESHOLD

# Configurar logging
logger = logging.getLogger(__name__)


class HumanApproval:
    """
    Gestor de aprobación humana para respuestas críticas.
    """
    
    def __init__(self, enabled: bool = None, confidence_threshold: float = None):
        """
        Inicializar gestor de aprobación humana.
        
        Args:
            enabled: Si habilitar HITL
            confidence_threshold: Threshold para aprobación automática
        """
        self.enabled = enabled if enabled is not None else HITL_ENABLED
        self.confidence_threshold = confidence_threshold or HITL_CONFIDENCE_THRESHOLD
        
        logger.info(f"HITL inicializado: enabled={self.enabled}, threshold={self.confidence_threshold}")
    
    def should_require_approval(
        self, 
        response: str, 
        confidence: Optional[float] = None,
        is_critical_topic: bool = False
    ) -> bool:
        """
        Determinar si la respuesta requiere aprobación humana.
        
        Args:
            response: Respuesta generada
            confidence: Confianza de la respuesta (0-1)
            is_critical_topic: Si es un tema crítico
        
        Returns:
            True si requiere aprobación, False si es automática
        """
        
        # Si HITL está deshabilitado, nunca requiere aprobación
        if not self.enabled:
            return False
        
        # Temas críticos siempre requieren aprobación
        if is_critical_topic:
            logger.debug("Tema crítico, requiere aprobación")
            return True
        
        # Si la confianza es menor al threshold, requiere aprobación
        if confidence is not None and confidence < self.confidence_threshold:
            logger.debug(f"Confianza baja ({confidence:.2f} < {self.confidence_threshold}), requiere aprobación")
            return True
        
        # Respuestas muy largas pueden requerir revisión
        if len(response) > 1000:
            logger.debug("Respuesta muy larga, requiere aprobación")
            return True
        
        # Por defecto, no requiere aprobación
        return False
    
    def request_approval(self, response: str, context: str = "") -> Tuple[bool, str]:
        """
        Solicitar aprobación humana para una respuesta.
        
        Args:
            response: Respuesta generada
            context: Contexto de la conversación
        
        Returns:
            Tuple de (aprobado, feedback)
        """
        
        print("\n" + "=" * 80)
        print("⚖️  APROBACIÓN HUMANA REQUERIDA")
        print("=" * 80)
        
        print("\n📝 Respuesta generada:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        print("\n¿Aprobar esta respuesta?")
        print("  [a] Aprobar")
        print("  [r] Rechazar")
        print("  [e] Editar")
        print("  [s] Saltar (no mostrar)")
        
        choice = input("\nTu decisión (a/r/e/s): ").strip().lower()
        
        if choice == "a":
            logger.info("Respuesta aprobada por humano")
            return True, ""
        
        elif choice == "r":
            feedback = input("Motivo del rechazo (opcional): ").strip()
            logger.info(f"Respuesta rechazada por humano: {feedback}")
            return False, feedback
        
        elif choice == "e":
            print("\nEdita la respuesta (presiona Enter dos veces para terminar):")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            
            edited_response = "\n".join(lines[:-1]).strip()  # Remover último Enter
            logger.info("Respuesta editada por humano")
            return True, edited_response
        
        elif choice == "s":
            logger.info("Aprobación saltada por humano")
            return False, "Saltado por usuario"
        
        else:
            logger.warning("Elección inválida, asumiendo rechazo")
            return False, "Elección inválida"
    
    def check_critical_topics(self, query: str) -> bool:
        """
        Verificar si la consulta es sobre un tema crítico.
        
        Args:
            query: Consulta del usuario
        
        Returns:
            True si es tema crítico
        """
        
        critical_keywords = [
            "demanda", "juicio", "cárcel", "detención",
            "muerte", "suicidio", "violencia", "abuso",
            "menor", "niño", "embarazo", "medicamento"
        ]
        
        query_lower = query.lower()
        
        is_critical = any(keyword in query_lower for keyword in critical_keywords)
        
        if is_critical:
            logger.debug(f"Tema crítico detectado en query: {query[:50]}...")
        
        return is_critical


# Instancia global para la aplicación
default_hitl = HumanApproval()


def get_default_hitl() -> HumanApproval:
    """
    Obtener gestor de aprobación humana por defecto.
    
    Returns:
        Instancia de HumanApproval
    """
    return default_hitl
```

---

## Fase 9: Testing

### Objetivo

Implementar tests unitarios para validar el funcionamiento del chatbot.

### Paso 9.1: Tests del Chatbot

```python
# tests/test_chatbot.py
"""
Tests unitarios para el Chatbot Legal.
"""

import pytest
from unittest.mock import Mock, patch
from src.chatbot import LegalChatbot
from src.memory import ChatMemory
from src.config import LLM_MODEL


class TestLegalChatbot:
    """Tests para la clase LegalChatbot."""
    
    @pytest.fixture
    def mock_llm(self):
        """LLM mockeado para tests."""
        mock = Mock()
        mock.invoke.return_value = Mock(content="Respuesta de prueba")
        return mock
    
    @pytest.fixture
    def chatbot(self, mock_llm):
        """Chatbot de prueba."""
        return LegalChatbot(llm=mock_llm)
    
    def test_initialization(self, chatbot):
        """Test de inicialización."""
        assert chatbot is not None
        assert chatbot.memory is not None
        assert chatbot.hitl is not None
    
    def test_generate_response_basic(self, chatbot, mock_llm):
        """Test de generación básica de respuesta."""
        response = chatbot.generate_response("¿Qué es una tutela?")
        
        assert response is not None
        assert "respuesta" in response
        assert "confidence" in response
        assert "requires_approval" in response
        
        # Verificar que el LLM fue llamado
        mock_llm.invoke.assert_called_once()
    
    def test_generate_response_with_memory(self, chatbot, mock_llm):
        """Test de generación con memoria."""
        # Agregar mensajes previos a la memoria
        chatbot.memory.add_message("Hola", "¡Hola! ¿En qué puedo ayudarte?")
        
        # Generar respuesta
        response = chatbot.generate_response("Tengo una pregunta legal")
        
        assert response is not None
        
        # La memoria debería haber sido usada
        assert chatbot.memory.turn_count == 1
    
    def test_generate_response_critical_topic(self, chatbot, mock_llm):
        """Test de respuesta para tema crítico."""
        response = chatbot.generate_response("¿Cómo interpongo una demanda por custodia?")
        
        assert response is not None
        assert response["requires_approval"] == True  # Tema crítico requiere aprobación
    
    def test_chat_basic(self, chatbot, mock_llm):
        """Test de chat básico."""
        response = chatbot.chat("¿Qué es el derecho de petición?")
        
        assert response is not None
        assert isinstance(response, dict)
        assert "respuesta" in response
    
    def test_chat_with_approval_required(self, chatbot, mock_llm):
        """Test de chat con aprobación requerida."""
        with patch.object(chatbot.hitl, 'request_approval') as mock_approval:
            mock_approval.return_value = (True, "")
            
            response = chatbot.chat("Necesito ayuda con un caso de divorcio")
            
            assert response is not None
            mock_approval.assert_called_once()
    
    def test_reset_conversation(self, chatbot, mock_llm):
        """Test de reset de conversación."""
        # Agregar mensajes
        chatbot.memory.add_message("Hola", "Hola")
        chatbot.memory.add_message("Adiós", "Adiós")
        
        # Resetear
        chatbot.reset_conversation()
        
        # Verificar que la memoria está limpia
        assert chatbot.memory.turn_count == 0
        assert len(chatbot.memory.get_history()) == 0
    
    def test_get_conversation_stats(self, chatbot, mock_llm):
        """Test de estadísticas de conversación."""
        # Agregar mensajes
        chatbot.memory.add_message("Pregunta 1", "Respuesta 1")
        chatbot.memory.add_message("Pregunta 2", "Respuesta 2")
        
        stats = chatbot.get_stats()
        
        assert stats["turn_count"] == 2
        assert stats["short_term_messages"] == 2
        assert stats["long_term_enabled"] == True


class TestChatMemory:
    """Tests para la clase ChatMemory."""
    
    @pytest.fixture
    def memory(self):
        """Memoria de prueba."""
        return ChatMemory()
    
    def test_add_message(self, memory):
        """Test de agregar mensaje."""
        memory.add_message("Hola", "¡Hola!")
        
        assert memory.turn_count == 1
        assert len(memory.get_history()) == 2  # Human + AI
    
    def test_get_history(self, memory):
        """Test de obtener historial."""
        memory.add_message("P1", "R1")
        memory.add_message("P2", "R2")
        
        history = memory.get_history()
        
        assert len(history) == 4
        assert "P1" in history[0].content
        assert "R1" in history[1].content
    
    def test_get_summary(self, memory):
        """Test de obtener resumen."""
        memory.add_message("Hola", "Hola")
        memory.add_message("¿Cómo estás?", "Bien")
        
        summary = memory.get_summary()
        
        assert summary is not None
        assert len(summary) > 0
    
    def test_clear(self, memory):
        """Test de limpiar memoria."""
        memory.add_message("P", "R")
        memory.clear()
        
        assert memory.turn_count == 0
        assert len(memory.get_history()) == 0
    
    def test_get_stats(self, memory):
        """Test de estadísticas."""
        memory.add_message("P1", "R1")
        memory.add_message("P2", "R2")
        
        stats = memory.get_stats()
        
        assert stats["turn_count"] == 2
        assert stats["short_term_messages"] == 4
        assert stats["long_term_enabled"] == True
```

### Paso 9.2: Ejecutar Tests

```bash
# Ejecutar tests
pytest tests/ -v

# Con coverage
pytest tests/ -v --cov=src --cov-report=html

# Ver reporte de coverage
open htmlcov/index.html  # Mac/Linux
start htmlcov\index.html  # Windows
```

---

## Fase 10: Empaquetado

### Objetivo

Empaquetar el proyecto con toda la documentación necesaria.

### Paso 10.1: README Final del Proyecto

```markdown
# 🤖 Chatbot Legal Básico

Chatbot conversacional para consultas legales básicas.

## Instalación

```bash
# Clonar o navegar al proyecto
cd proyectos/01-chatbot-legal-basico

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu API Key de Google Gemini
```

## Uso

```bash
# Ejecutar chatbot
python main.py

# O usar como módulo
from src.chatbot import LegalChatbot
chatbot = LegalChatbot()
response = chatbot.chat("¿Qué es una tutela?")
print(response["respuesta"])
```

## Tests

```bash
# Ejecutar tests
pytest tests/ -v

# Con coverage
pytest tests/ -v --cov=src
```

## Estructura

```
01-chatbot-legal-basico/
├── src/
│   ├── config.py          # Configuración
│   ├── llm.py             # LLM
│   ├── memory.py          # Memoria
│   ├── human_in_loop.py   # HITL
│   └── chatbot.py         # Chatbot principal
├── tests/
│   ├── test_chatbot.py
│   └── test_memory.py
├── docs/
│   └── README.md
├── .env
├── .env.example
├── requirements.txt
└── main.py
```

## Características

- ✅ Consultas legales básicas
- ✅ Memoria de conversación
- ✅ Aprobación humana para temas críticos
- ✅ Logging completo
- ✅ Tests unitarios

## Próximo Proyecto

➡️ Proyecto 2: RAG Documental Legal
```

### Paso 10.2: Archivo `.env.example`

```bash
# .env.example
# Copiar a .env y configurar valores reales

# Google Gemini API Key
GOOGLE_API_KEY=tu_api_key_aqui

# Configuración de la aplicación
APP_NAME=Chatbot Legal Básico
APP_VERSION=1.0.0
LOG_LEVEL=INFO

# Configuración del LLM
LLM_MODEL=gemini-2.0-flash-exp
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000

# Memoria
MEMORY_SHORT_TERM_K=5
MEMORY_LONG_TERM_ENABLED=true

# Human in the Loop
HITL_ENABLED=true
HITL_CONFIDENCE_THRESHOLD=0.7
```

---

## ✅ Checklist de Completación

- [ ] Fase 1: Importación y Configuración ✅
- [ ] Fase 2: Invocar Modelo ✅
- [ ] Fase 3: Chat Prompt Template ⏳
- [ ] Fase 4: System Prompt ⏳
- [ ] Fase 5: Response + Parsing ⏳
- [ ] Fase 6: Memoria Corto Plazo ✅
- [ ] Fase 7: Memoria Largo Plazo ✅
- [ ] Fase 8: Human in the Loop ✅
- [ ] Fase 9: Testing ✅
- [ ] Fase 10: Empaquetado ✅

---

*Documentación creada: 2026-03-29*
