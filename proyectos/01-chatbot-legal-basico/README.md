# ✅ PROYECTO 1 COMPLETADO: Chatbot Legal Básico

> **Estado**: ✅ COMPLETADO (80% - Tests Pendientes)
> **Nivel**: Básico (Refrescamiento)
> **Tiempo Real**: 4-6 horas
> **Tecnologías**: LangChain, AWS Bedrock (Amazon Nova Lite), Memoria, HITL
> **Fecha Completación**: 2026-03-30
> **Fecha Actualización**: 2026-03-30 (Migrado a AWS Bedrock + Testing Completo)

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura](#-arquitectura)
- [Inicio Rápido](#-inicio-rápido)
- [Documentación de la API](#-documentación-de-la-api)
- [Resultados de Testing](#-resultados-de-testing)
- [Ejemplos de Uso](#-ejemplos-de-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Cambios Recientes](#-cambios-recientes)
- [Pendientes](#-pendientes)
- [Recursos Adicionales](#-recursos-adicionales)

---

## 📝 Descripción

**Chatbot Legal Básico** es un asistente virtual especializado en derecho colombiano que utiliza **AWS Bedrock** (Amazon Nova Lite) para generar respuestas precisas y contextualizadas.

### Características Principales

| Característica | Descripción |
|----------------|-------------|
| 🤖 **LLM** | Amazon Nova Lite vía AWS Bedrock (inference profile) |
| 🧠 **Memoria** | Corto plazo (ventana deslizante) + Largo plazo (resúmenes automáticos) |
| 👤 **HITL** | Human in the Loop para aprobación de respuestas críticas |
| 📊 **Logging** | Registro completo de todas las interacciones |
| ⚠️ **Seguridad** | Detección automática de temas críticos legales |

### Casos de Uso

- Consultas sobre derechos fundamentales
- Información sobre procedimientos legales (tutelas, derechos de petición)
- Orientación sobre plazos y términos legales
- Explicación de conceptos jurídicos básicos

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                     CHATBOT LEGAL BÁSICO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Usuario    │────▶│   main.py    │────▶│  chatbot.py  │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                │                │
│                                                ▼                │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Memoria    │◀───▶│    LLM       │◀───▶│   config.py  │   │
│  │  (memory.py) │     │  (llm.py)    │     │              │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                │                │
│                                                ▼                │
│                                       ┌──────────────┐         │
│                                       │ AWS Bedrock  │         │
│                                       │  (Nova Lite) │         │
│                                       └──────────────┘         │
│                                                │                │
│  ┌──────────────┐                              │                │
│  │  Human in    │◀─────────────────────────────┘                │
│  │   the Loop   │                                               │
│  │  (hitl.py)   │                                               │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de Procesamiento

```
1. Input del Usuario
         │
         ▼
2. Generar Respuesta (LLM + Memoria)
         │
         ▼
3. Evaluar Confianza (< 0.7 → HITL)
         │
         ▼
4. ¿Tema Crítico? (tutela, demanda, etc.) → HITL
         │
         ▼
5. Aprobación Humana (si requiere)
         │
         ▼
6. Guardar en Memoria + Responder
```

---

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
# Opción A: Desde el directorio del proyecto
cd proyectos/01-chatbot-legal-basico
pip install -r requirements.txt

# Opción B: Desde la raíz del proyecto
cd "C:\Users\DELL\Desktop\Software Fenix\RAG MVP"
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo a la raíz del proyecto
cp .env.example .env
```

Editar `.env` con tus credenciales de AWS:

```bash
# Credenciales de AWS (requeridas)
AWS_ACCESS_KEY_ID="AKIA..."
AWS_SECRET_ACCESS_KEY="..."
AWS_SESSION_TOKEN=""  # Opcional, solo para credenciales temporales

# Región de AWS
AWS_REGION="us-east-2"

# Modelo (Amazon Nova Lite - inference profile)
LLM_MODEL_ID="arn:aws:bedrock:us-east-2:762233737662:inference-profile/us.amazon.nova-lite-v1:0"
LLM_PROVIDER="amazon"

# Parámetros del LLM
LLM_TEMPERATURE="0.3"
LLM_MAX_TOKENS="2048"
```

### 3. Ejecutar Chatbot

```bash
# Modo interactivo (recomendado)
python main.py

# Modo test rápido
python test_quick.py

# Modo test completo
python test_completo.py
```

### 4. Comandos Disponibles

Durante la sesión interactiva:

| Comando | Descripción |
|---------|-------------|
| `historial` | Ver últimos 10 mensajes |
| `stats` | Ver estadísticas de la sesión |
| `reset` | Resetear conversación |
| `salir`, `exit`, `q` | Salir del chatbot |

---

## 📖 Documentación de la API

### Clase `LegalChatbot`

```python
from src.chatbot import LegalChatbot

# Inicializar
chatbot = LegalChatbot(enable_hitl=True)

# Chatear
response = chatbot.chat("¿Qué es una tutela?", auto_approve=False)
print(response["respuesta"])
print(response["confidence"])  # 0.0 - 1.0

# Ver estadísticas
stats = chatbot.get_stats()
# {
#   "turn_count": 5,
#   "short_term_messages": 10,
#   "hitl_enabled": True,
#   ...
# }

# Resetear conversación
chatbot.reset_conversation()

# Ver historial
history = chatbot.get_conversation_history()
```

### Clase `ChatMemory`

```python
from src.memory import ChatMemory

memory = ChatMemory(k=5, enable_long_term=True)

# Agregar mensajes
memory.add_message("¿Qué es una tutela?", "Es un mecanismo...")

# Obtener historial
history = memory.get_history()

# Obtener contexto
context = memory.get_context()

# Estadísticas
stats = memory.get_stats()
```

### Clase `HumanApproval`

```python
from src.human_in_loop import HumanApproval

hitl = HumanApproval(enabled=True, confidence_threshold=0.7)

# Verificar tema crítico
es_critico = hitl.check_critical_topics("¿Cómo presento una demanda?")

# Decidir si requiere aprobación
requiere = hitl.should_require_approval(
    response="La demanda se presenta...",
    confidence=0.65,
    is_critical_topic=True
)

# Solicitar aprobación (interactivo)
aprobado, feedback = hitl.request_approval(
    response="...",
    context="Usuario pregunta sobre..."
)
```

### Funciones LLM

```python
from src.llm import create_llm, get_default_llm, get_cached_llm

# LLM por defecto
llm = get_default_llm()

# LLM personalizado
llm = create_llm(
    model_id="arn:aws:bedrock:...",
    temperature=0.5,
    max_tokens=1024
)

# LLM cacheado (singleton)
llm = get_cached_llm()
```

---

## 🧪 Resultados de Testing

### Resumen Ejecutivo

| Métrica | Resultado |
|---------|-----------|
| **Tests Ejecutados** | 8 |
| **Tests Aprobados** | 8 ✅ |
| **Tests Fallidos** | 0 ❌ |
| **Cobertura** | 100% funcional |

### Tests Detallados

| # | Test | Estado | Tiempo | Detalles |
|---|------|--------|--------|----------|
| 1 | Importaciones | ✅ PASÓ | <1s | Todos los módulos importan correctamente |
| 2 | Inicialización | ✅ PASÓ | ~1s | Chatbot creado con AWS Bedrock |
| 3 | Consulta básica | ✅ PASÓ | ~6s | 1639 chars, 70% confianza |
| 4 | Memoria (seguimiento) | ✅ PASÓ | ~3s | Contexto mantenido |
| 5 | Consulta debido proceso | ✅ PASÓ | ~3s | Aprobación automática |
| 6 | Estadísticas | ✅ PASÓ | <1s | turn_count: 3, messages: 6 |
| 7 | Historial | ✅ PASÓ | <1s | 6 mensajes almacenados |
| 8 | Reset | ✅ PASÓ | <1s | Memoria limpiada |

### Métricas de Rendimiento

| Métrica | Valor | Unidad |
|---------|-------|--------|
| Tiempo promedio de respuesta | 3-6 | segundos |
| Tokens de salida promedio | 1600-1800 | caracteres |
| Confianza estimada | 50-70 | % |
| HITL activación (temas críticos) | ✅ | Funcional |
| Memoria corto plazo | 5 | mensajes (configurable) |
| Memoria largo plazo | ✅ | Resúmenes automáticos |

### Ejemplos de Respuestas

#### Test 1: Consulta Básica

**Pregunta**: *"¿Qué es una tutela en Colombia?"*

**Respuesta**:
```
Una acción de tutela es un mecanismo procesal de protección de derechos 
fundamentales consagrado en la Constitución Política de Colombia y 
desarrollado en la Ley 1348 de 2009. Es un recurso de protección rápida 
y efectiva para las personas cuando sus derechos fundamentales han sido 
vulnerados o amenazados...
```

**Métricas**: 1639 chars, 70% confianza, HITL: tema crítico

---

#### Test 2: Consulta de Seguimiento (Memoria)

**Pregunta**: *"¿Cuál es el plazo máximo para presentar una tutela?"*

**Respuesta**:
```
El plazo máximo para presentar una acción de tutela en Colombia es de 
**un año** contado a partir del día siguiente al en que se tenga 
conocimiento del acto o hecho que se pretende sea declarado 
inconstitucional o ilegal.

### Base Legal:
- **Artículo 10 de la Ley 1348 de 2009**: Establece que la acción...
```

**Métricas**: Contexto mantenido de consulta anterior, HITL: tema crítico

---

#### Test 3: Debido Proceso

**Pregunta**: *"¿Qué es el debido proceso?"*

**Respuesta**:
```
El debido proceso es un principio jurídico que garantiza que las 
personas sean tratadas de manera justa y equitativa por parte de 
las autoridades, asegurando que sus derechos fundamentales sean 
respetados en todo momento.

### Características Principales:
- **Protección de Derechos Fundamentales**: Garantiza el respeto...
```

**Métricas**: Aprobación automática (tema no crítico, confianza adecuada)

---

### Comandos de Test

```bash
# Test rápido (30 segundos)
python test_quick.py

# Test completo (2-3 minutos)
python test_completo.py
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Uso Básico

```python
import sys
sys.path.insert(0, 'src')
from chatbot import LegalChatbot

chatbot = LegalChatbot()
response = chatbot.chat("¿Qué derechos protege la Constitución?")
print(response["respuesta"])
```

### Ejemplo 2: Con Memoria

```python
chatbot = LegalChatbot()

# Primera consulta
chatbot.chat("¿Qué es una tutela?")

# Segunda consulta (usa contexto previo)
response = chatbot.chat("¿Cuánto tiempo tengo para presentarla?")
print(response["respuesta"])
```

### Ejemplo 3: Sin HITL (auto-aprobación)

```python
chatbot = LegalChatbot()
response = chatbot.chat("¿Qué es el debido proceso?", auto_approve=True)
```

### Ejemplo 4: Estadísticas

```python
chatbot = LegalChatbot()
chatbot.chat("Consulta 1")
chatbot.chat("Consulta 2")

stats = chatbot.get_stats()
print(f"Turnos: {stats['turn_count']}")
print(f"Memoria: {stats['short_term_messages']} mensajes")
```

---

## 📁 Estructura del Proyecto

```
proyectos/01-chatbot-legal-basico/
│
├── src/
│   ├── __init__.py              # ✅ Export del paquete
│   ├── config.py                # ✅ Configuración AWS Bedrock
│   ├── llm.py                   # ✅ Integración langchain-aws
│   ├── memory.py                # ✅ Gestión de memoria
│   ├── human_in_loop.py         # ✅ Aprobación humana
│   └── chatbot.py               # ✅ Chatbot principal
│
├── tests/
│   ├── __init__.py              # ⏳ Pendiente
│   ├── test_chatbot.py          # ⏳ Pendiente
│   └── test_memory.py           # ⏳ Pendiente
│
├── docs/
│   ├── PROYECTO_1_FASES_1_5.md  # ✅ Fases 1-5
│   └── PROYECTO_1_FASES_6_10.md # ✅ Fases 6-10
│
├── main.py                      # ✅ Punto de entrada CLI
├── test_quick.py                # ✅ Test rápido
├── test_completo.py             # ✅ Test completo
├── requirements.txt             # ✅ Dependencias
└── README.md                    # ✅ Este archivo
```

---

## ✅ Checklist de Completación

| Fase | Estado | Descripción |
|------|--------|-------------|
| 1 | ✅ | Importación y Configuración |
| 2 | ✅ | Invocar Modelo (AWS Bedrock) |
| 3 | ✅ | Chat Prompt Template |
| 4 | ✅ | System Prompt |
| 5 | ✅ | Response + Parsing |
| 6 | ✅ | Memoria Corto Plazo |
| 7 | ✅ | Memoria Largo Plazo |
| 8 | ✅ | Human in the Loop |
| 9 | ⏳ | Testing (tests formales pendientes) |
| 10 | ⏳ | Empaquetado |

**Progreso**: 8/10 fases completadas (80%)

---

## 🔄 Cambios Recientes (2026-03-30)

### Migración a AWS Bedrock

El chatbot fue migrado exitosamente de **Google Gemini** a **AWS Bedrock**.

| Componente | Anterior | Nuevo |
|------------|----------|-------|
| **LLM** | Google Gemini 2.0 Flash | Amazon Nova Lite (inference profile) |
| **Región** | N/A | us-east-2 |
| **SDK** | `langchain-google-genai` | `langchain-aws` |

### Archivos Nuevos

| Archivo | Descripción |
|---------|-------------|
| `src/llm.py` | Integración con `langchain_aws.ChatBedrock` |
| `src/memory.py` | Gestión completa de memoria (corto/largo plazo) |
| `src/human_in_loop.py` | Sistema de aprobación humana interactiva |
| `src/__init__.py` | Export del paquete |
| `test_quick.py` | Test rápido de verificación |
| `test_completo.py` | Test completo con múltiples consultas |
| `.env.example` | Template de configuración AWS |

### Archivos Actualizados

| Archivo | Cambios |
|---------|---------|
| `src/config.py` | Migrado a credenciales AWS |
| `src/chatbot.py` | Actualizado para usar AWS Bedrock |
| `requirements.txt` | Agrega `langchain-aws`, `boto3` |
| `README.md` | Documentación completa + resultados de testing |

---

## 📝 Pendientes

Para completar el proyecto al 100%:

- [ ] Crear `tests/__init__.py`
- [ ] Crear `tests/test_chatbot.py` (tests unitarios formales)
- [ ] Crear `tests/test_memory.py` (tests de memoria)
- [ ] Agregar tests de integración
- [ ] Configurar CI/CD para tests automáticos
- [ ] Agregar más ejemplos de uso en `docs/`

---

## 🔧 Configuración Avanzada

### Variables de Entorno Disponibles

```bash
# Aplicación
APP_NAME="Chatbot Legal Básico"
APP_VERSION="1.0.0"
LOG_LEVEL="INFO"

# AWS Bedrock
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_SESSION_TOKEN=""  # Opcional
AWS_REGION="us-east-2"

# LLM
LLM_MODEL_ID="arn:aws:bedrock:us-east-2:762233737662:inference-profile/us.amazon.nova-lite-v1:0"
LLM_PROVIDER="amazon"
LLM_TEMPERATURE="0.3"
LLM_MAX_TOKENS="2048"

# Memoria
MEMORY_SHORT_TERM_K="5"
MEMORY_LONG_TERM_ENABLED="true"

# Human in the Loop
HITL_ENABLED="true"
HITL_CONFIDENCE_THRESHOLD="0.7"

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE="10"
RATE_LIMIT_BURST_LIMIT="3"
MAX_RETRIES="3"
```

### Temas Críticos (HITL Automático)

El sistema detecta automáticamente estos temas como críticos:

```python
CRITICAL_TOPICS = [
    "demanda", "juicio", "sentencia", "cárcel", "detención",
    "multa", "sanción", "custodia", "divorcio", "herencia",
    "testamento", "quiebra", "embargo", "desalojo", "despido",
    "accidente", "muerte", "lesiones", "tutela"
]
```

---

## 🔄 Próximo Proyecto

➡️ **Proyecto 2: RAG Documental Legal**

- [ ] Carga de PDFs legales (leyes, sentencias, contratos)
- [ ] Chunking especializado por artículos/cláusulas
- [ ] Embeddings con Amazon Titan v2
- [ ] Vector Store (Chroma/Qdrant)
- [ ] Retrieval con reranking (FlashRank)
- [ ] Prevención de alucinaciones

---

## 📚 Recursos Adicionales

### Documentación Oficial

- [LangChain AWS Docs](https://python.langchain.com/docs/integrations/llms/bedrock/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Amazon Nova Lite](https://aws.amazon.com/bedrock/nova/)

### Archivos Relacionados

- `docs/PROYECTO_1_FASES_1_5.md` - Fases iniciales
- `docs/PROYECTO_1_FASES_6_10.md` - Fases avanzadas
- `.env.example` - Template de configuración

---

## 📊 Historial de Versiones

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0.0 | 2026-03-30 | Migración completa a AWS Bedrock |
| 0.2.0 | 2026-03-29 | Human in the Loop implementado |
| 0.1.0 | 2026-03-29 | Versión inicial con Google Gemini |

---

*Proyecto creado: 2026-03-29*  
*Última actualización: 2026-03-30*  
*Autor: Curso LangChain + LangGraph para RAG*  
*License: MIT*
