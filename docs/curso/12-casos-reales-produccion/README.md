# Fase Adicional: Casos Reales y Producción

> **Objetivo**: Implementar patrones de agentes avanzados con código de producción y casos de uso reales  
> **Basado en**: Documentación oficial LangChain/LangGraph + Patrones de la industria  
> **Estado**: ✅ COMPLETADO

---

## 📋 Índice de la Fase

### Parte 1: Patrones de Agentes Avanzados
1. [12.1 - Prompt Chaining](#121-prompt-chaining)
2. [12.2 - Parallelization](#122-parallelization)
3. [12.3 - Routing](#123-routing)
4. [12.4 - Orchestrator-Workers](#124-orchestrator-workers)
5. [12.5 - Evaluator-Optimizer](#125-evaluator-optimizer)
6. [12.6 - Multi-Agent Handoff](#126-multi-agent-handoff)

### Parte 2: Casos de Uso Reales
7. [12.7 - Asistente Legal Corporativo](#127-asistente-legal-corporativo)
8. [12.8 - Clasificador y Enrutador de Documentos](#128-clasificador-y-enrutador-de-documentos)
9. [12.9 - Agente de Due Diligence](#129-agente-de-due-diligence)
10. [12.10 - Sistema de Compliance](#1210-sistema-de-compliance)

### Parte 3: Cuándo y Cómo Combinar Patrones
11. [Guía de Selección de Patrones](#guía-de-selección-de-patrones)
12. [Combinación de Patrones](#combinación-de-patrones)

---

## Parte 1: Patrones de Agentes Avanzados

### 12.1 - Prompt Chaining

**Qué es**: Descomponer una tarea compleja en múltiples pasos secuenciales.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Paso 1    │ →  │   Paso 2    │ →  │   Paso 3    │ →  │   Resultado │
│  Extraer    │    │  Analizar   │    │  Redactar   │    │    Final    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Cuándo usar**:
- ✅ Tareas que requieren múltiples pasos lógicos
- ✅ Cada paso depende del resultado anterior
- ✅ Necesitas control total del flujo

**Ejemplo Real**: Análisis de contratos
```python
# Flujo:
1. Extraer cláusulas → 2. Identificar riesgos → 3. Generar resumen → 4. Redactar informe
```

---

### 12.2 - Parallelization

**Qué es**: Ejecutar múltiples tareas simultáneamente para reducir latencia.

```
                    ┌─────────────┐
                    │   Input     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Worker 1 │     │ Worker 2 │     │ Worker 3 │
   │ Análisis │     │ Búsqueda │     │ Validación│
   └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                │
         └────────────────┴────────────────┘
                          ↓
                    ┌──────────┐
                    │ Consolidar│
                    └──────────┘
```

**Cuándo usar**:
- ✅ Tareas independientes entre sí
- ✅ Necesitas reducir latencia
- ✅ Tienes recursos disponibles

**Ejemplo Real**: Búsqueda multi-fuente
```python
# Ejecutar en paralelo:
- Búsqueda en vector store
- Búsqueda en base de datos SQL
- Búsqueda en web
→ Consolidar resultados
```

---

### 12.3 - Routing

**Qué es**: Clasificar input y dirigir a diferentes rutas especializadas.

```
                    ┌─────────────┐
                    │   Input     │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │   Router    │
                    │ (Clasifica) │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  Ruta A  │     │  Ruta B  │     │  Ruta C  │
   │ Legal    │     │ Laboral  │     │ Penal    │
   └──────────┘     └──────────┘     └──────────┘
```

**Cuándo usar**:
- ✅ Diferentes tipos de input requieren diferentes tratamientos
- ✅ Tienes especialistas por categoría
- ✅ Quieres optimizar costos/tiempo

**Ejemplo Real**: Clasificación de consultas legales
```python
# Router clasifica:
- "¿Me pueden despedir?" → Agente Laboral
- "¿Cómo creo una empresa?" → Agente Corporativo
- "¿Qué pena tiene esto?" → Agente Penal
```

---

### 12.4 - Orchestrator-Workers

**Qué es**: Un orquestador central coordina múltiples workers especializados.

```
                    ┌─────────────────┐
                    │  Orquestador    │
                    │  (Coordina)     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ↓                   ↓                   ↓
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │  Worker 1  │     │  Worker 2  │     │  Worker 3  │
   │ Búsqueda   │     │  Análisis  │     │ Redacción  │
   └────────────┘     └────────────┘     └────────────┘
```

**Cuándo usar**:
- ✅ Tarea compleja que requiere múltiples habilidades
- ✅ Workers pueden reutilizarse
- ✅ Necesitas flexibilidad

**Ejemplo Real**: Investigación legal completa
```python
Orquestador recibe: "Investiga jurisprudencia sobre tutela y salud"

Orquestador coordina:
1. Worker Búsqueda → Encuentra fallos relevantes
2. Worker Análisis → Extrae principios jurídicos
3. Worker Redacción → Genera informe consolidado
```

---

### 12.5 - Evaluator-Optimizer

**Qué es**: Un agente genera, otro evalúa y critica, se itera hasta alcanzar calidad.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Generator  │ →  │  Evaluator  │ →  │   ¿Calidad? │
│  (Crea)     │    │  (Critica)  │    │   (Decide)  │
└─────────────┘    └─────────────┘    └──────┬──────┘
       ↑                                      │
       │         ┌─────────────┐              │
       └─────────│  Optimizer  │ ←─────── No ─┘
                 │  (Mejora)   │
                 └─────────────┘
                        ↓
                      Sí → Resultado Final
```

**Cuándo usar**:
- ✅ Calidad es crítica
- ✅ Criterios de calidad son claros
- ✅ Tienes tiempo para iterar

**Ejemplo Real**: Redacción de documentos legales
```python
1. Generator → Redacta demanda
2. Evaluator → Evalúa: "Faltan fundamentos jurídicos"
3. Optimizer → Agrega fundamentos
4. Evaluator → "Ahora cumple"
5. Resultado final
```

---

### 12.6 - Multi-Agent Handoff

**Qué es**: Agentes se transfieren la conversación según especialización necesaria.

```
┌─────────────┐
│  Usuario    │
└──────┬──────┘
       ↓
┌─────────────┐
│  Agente A   │ ← "Tengo una pregunta sobre..."
│  General    │
└──────┬──────┘
       │ Handoff (detecta especialización necesaria)
       ↓
┌─────────────┐
│  Agente B   │ ← "Como especialista en..."
│  Especialista│
└─────────────┘
```

**Cuándo usar**:
- ✅ Dominio con múltiples especialidades
- ✅ Cada agente tiene conocimiento especializado
- ✅ Usuario no necesita saber la especialización

**Ejemplo Real**: Firma de abogados
```python
Cliente → Agente Recepción → Agente Laboral → Agente Litigios → Informe Final
```

---

## Parte 2: Casos de Uso Reales

### 12.7 - Asistente Legal Corporativo

**Descripción**: Asistente para consultas legales de empleados de una empresa.

**Arquitectura**:
```
Router Pattern + Multi-Agent Handoff

Usuario → Router (clasifica consulta) → Agente Especialista → Respuesta
```

**Implementación**:
```python
# Casos de uso:
1. Consultas laborales → Agente Laboral
2. Consultas contractuales → Agente Contratos
3. Consultas de compliance → Agente Compliance
4. Consultas tributarias → Agente Tributario
```

**Código**: Ver `src/course_examples/casos_reales/01_asistente_legal_corporativo.py`

---

### 12.8 - Clasificador y Enrutador de Documentos

**Descripción**: Clasifica documentos legales y los enruta al proceso adecuado.

**Arquitectura**:
```
Parallelization + Routing

Documento → [Extraer texto, Analizar formato, Identificar tipo] → Router → Proceso
```

**Implementación**:
```python
# Tipos de documentos:
- Tutelas → Proceso: Extraer derechos vulnerados
- Contratos → Proceso: Extraer cláusulas y obligaciones
- Sentencias → Proceso: Extraer decisión y fundamentos
- Demandas → Proceso: Extraer pretensiones y hechos
```

**Código**: Ver `src/course_examples/casos_reales/02_clasificador_documentos.py`

---

### 12.9 - Agente de Due Diligence

**Descripción**: Realiza due diligence legal de empresas o personas.

**Arquitectura**:
```
Orchestrator-Workers + Evaluator-Optimizer

Orchestrador coordina:
1. Worker Búsqueda → Busca antecedentes
2. Worker Análisis → Identifica riesgos
3. Worker Verificación → Valida información
4. Evaluator → Evalúa completitud
5. Optimizer → Completa información faltante
```

**Implementación**:
```python
# Check de due diligence:
- Antecedentes judiciales
- Obligaciones laborales
- Contratos vigentes
- Litigios en curso
- Compliance regulatorio
```

**Código**: Ver `src/course_examples/casos_reales/03_due_diligence.py`

---

### 12.10 - Sistema de Compliance

**Descripción**: Monitorea y verifica cumplimiento normativo.

**Arquitectura**:
```
Evaluator-Optimizer + Prompt Chaining

Flujo:
1. Extraer normativa aplicable
2. Evaluar cumplimiento actual
3. Identificar brechas
4. Generar plan de acción
5. Optimizar hasta alcanzar 100% compliance
```

**Implementación**:
```python
# Áreas de compliance:
- Laboral (contratos, nómina, prestaciones)
- Tributario (declaraciones, pagos)
- Contractual (obligaciones, vencimientos)
- Regulatorio (licencias, permisos)
```

**Código**: Ver `src/course_examples/casos_reales/04_compliance.py`

---

## Parte 3: Guía de Selección de Patrones

### Tabla de Decisión

| Escenario | Patrón Recomendado | Alternativa |
|-----------|-------------------|-------------|
| **Tarea secuencial simple** | Prompt Chaining | - |
| **Múltiples fuentes independientes** | Parallelization | - |
| **Diferentes tipos de input** | Routing | Multi-Agent Handoff |
| **Tarea compleja multi-habilidad** | Orchestrator-Workers | Multi-Agent |
| **Calidad crítica** | Evaluator-Optimizer | - |
| **Especialización necesaria** | Multi-Agent Handoff | Routing |
| **Recursos limitados** | Sequential | Parallelization |
| **Tiempo crítico** | Parallelization | Routing |

### Combinación de Patrones

**Patrón Compuesto 1**: Router + Parallelization
```python
# Caso: Consulta compleja que requiere múltiples especialistas
Usuario → Router → [Agente A + Agente B + Agente C] → Consolidar → Respuesta
```

**Patrón Compuesto 2**: Orchestrator + Evaluator-Optimizer
```python
# Caso: Documento legal que requiere máxima calidad
Orchestrador → Workers generan → Evaluator evalúa → Optimizer mejora → Iterar
```

**Patrón Compuesto 3**: Multi-Agent Handoff + Prompt Chaining
```python
# Caso: Proceso legal de múltiples etapas
Agente 1 (Paso 1) → Handoff → Agente 2 (Paso 2) → Handoff → Agente 3 (Paso 3)
```

---

## 📊 Matriz de Patrones por Industria

### Legal

| Caso de Uso | Patrón | Complejidad |
|-------------|--------|-------------|
| Consulta simple | Prompt Chaining | Baja |
| Clasificación de documentos | Routing | Media |
| Due diligence | Orchestrator-Workers | Alta |
| Litigio complejo | Multi-Agent Handoff | Muy Alta |
| Revisión de contratos | Evaluator-Optimizer | Alta |

### Finanzas

| Caso de Uso | Patrón | Complejidad |
|-------------|--------|-------------|
| Análisis de riesgo | Parallelization | Media |
| Compliance | Evaluator-Optimizer | Alta |
| Atención al cliente | Routing | Baja |
| Reportes regulatorios | Prompt Chaining | Media |

### Salud

| Caso de Uso | Patrón | Complejidad |
|-------------|--------|-------------|
| Diagnóstico asistido | Evaluator-Optimizer | Muy Alta |
| Triaje de pacientes | Routing | Media |
| Historial clínico | Prompt Chaining | Media |
| Coordinación de especialistas | Multi-Agent Handoff | Alta |

---

## 🎯 Ejercicios Prácticos

### Ejercicio 1: Diseña la Arquitectura

**Escenario**: Firma de abogados necesita sistema para:
1. Recibir consultas de clientes
2. Clasificar por área legal
3. Asignar al abogado especializado
4. Generar respuesta
5. Evaluar satisfacción

**Tu tarea**: Dibuja la arquitectura y justifica patrones seleccionados.

---

### Ejercicio 2: Implementa Router Legal

**Requerimiento**: Crear router que clasifique consultas en:
- Laboral (despidos, contratos, prestaciones)
- Corporativo (empresas, contratos comerciales)
- Penal (delitos, defensas)
- Civil (familia, propiedades, sucesiones)

**Tu tarea**: Implementa el router con LLM y define los prompts para cada ruta.

---

### Ejercicio 3: Optimiza para Producción

**Escenario**: Sistema actual tiene:
- Latencia: 30 segundos
- Costo: $0.50 por consulta
- Calidad: 85% satisfacción

**Meta**: 
- Latencia: <10 segundos
- Costo: <$0.20 por consulta
- Calidad: >95% satisfacción

**Tu tarea**: Propón optimizaciones usando patrones aprendidos.

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [LangGraph Workflows](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Multi-Agent Systems](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Context Engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)

### Artículos de la Industria
- [Production-Ready AI Agents: 8 Patterns](https://towardsai.net/p/machine-learning/production-ready-ai-agents-8-patterns-that-actually-work)
- [7 Must-Know Agentic AI Design Patterns](https://machinelearningmastery.com/7-must-know-agentic-ai-design-patterns/)
- [Zero to One: Learning Agentic Patterns](https://www.philschmid.de/agentic-pattern)

---

*Siguiente: Implementación de código en `src/course_examples/casos_reales/`*
