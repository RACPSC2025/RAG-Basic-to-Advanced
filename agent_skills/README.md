# 🎓 Agent Skills - RAG Legal

> **Skills especializados para construir agentes legales con LangChain, LangGraph y Deep Agents**

Este directorio contiene **agent skills** oficiales de LangChain adaptados al dominio legal colombiano.

---

## 📚 Skills Disponibles

### Skills de LangChain

| Skill | Descripción | Estado |
|-------|-------------|--------|
| [`langchain-rag-legal`](./langchain-rag-legal/SKILL.md) | RAG especializado para documentos legales | ✅ Completado |
| `langchain-fundamentals` | Fundamentos de LangChain para legal | ⏳ Pendiente |
| `langchain-middleware` | Middleware para flujos legales | ⏳ Pendiente |

### Skills de LangGraph

| Skill | Descripción | Estado |
|-------|-------------|--------|
| [`langgraph-fundamentals-legal`](./langgraph-fundamentals-legal/SKILL.md) | Fundamentos de LangGraph para legal | ✅ Completado |
| [`langgraph-human-in-the-loop-legal`](./langgraph-human-in-the-loop-legal/SKILL.md) | HITL para decisiones legales críticas | ✅ Completado |
| `langgraph-persistence` | Persistencia de estado legal | ⏳ Pendiente |

### Skills de Deep Agents

| Skill | Descripción | Estado |
|-------|-------------|--------|
| `deep-agents-core` | Arquitectura base de agentes | ⏳ Pendiente |
| `deep-agents-memory` | Memoria para agentes legales | ⏳ Pendiente |
| `deep-agents-orchestration` | Orquestación multi-agente | ⏳ Pendiente |

---

## 🚀 Cómo Usar los Skills

### Opción 1: Con Gemini CLI

```bash
# Navegar al proyecto
cd "C:\Users\DELL\Desktop\Software Fenix\RAG MVP"

# Agregar skill específico
npx skills add langchain-ai/langchain-skills --skill langchain-rag

# O agregar todos los skills
npx skills add langchain-ai/langchain-skills --skill '*' --yes
```

### Opción 2: Con Claude Code

```bash
# Como plugin
/plugin marketplace add langchain-ai/langchain-skills
/plugin install langchain-skills@langchain-skills

# O manualmente
./install.sh --global
```

### Opción 3: Manual (Recomendado para este proyecto)

Los skills ya están en `agent_skills/` y listos para usar.

**Estructura de un Skill**:
```
agent_skills/
└── nombre-del-skill/
    ├── SKILL.md           # Requerido: Metadata e instrucciones
    ├── scripts/           # Opcional: Scripts ejecutables
    ├── references/        # Opcional: Documentación estática
    └── assets/            # Opcional: Templates y recursos
```

---

## 📖 Skill: langchain-rag-legal

**Propósito**: Construir sistemas RAG para documentos legales con máxima precisión.

**Cubre**:
- Carga de PDFs legales (leyes, sentencias, contratos)
- Chunking especializado por artículos/cláusulas
- Embeddings con Google Gemini
- Vector Store con Qdrant
- Retrieval con reranking (FlashRank)
- Prevención de alucinaciones

**Cuándo usar**:
- Q&A sobre jurisprudencia
- Búsqueda en contratos
- Análisis de normativa

**Ver más**: [`langchain-rag-legal/SKILL.md`](./langchain-rag-legal/SKILL.md)

---

## 📖 Skill: langgraph-fundamentals-legal

**Propósito**: Construir flujos de trabajo legales con LangGraph.

**Cubre**:
- StateGraph para procesos legales
- Nodes y edges
- Conditional routing
- State management
- Patrones: clasificador, multi-fase, bucle de refinamiento

**Cuándo usar**:
- Flujos de aprobación de documentos
- Análisis multi-etapa de contratos
- Sistemas de clasificación automática

**Ver más**: [`langgraph-fundamentals-legal/SKILL.md`](./langgraph-fundamentals-legal/SKILL.md)

---

## 📖 Skill: langgraph-human-in-the-loop-legal

**Propósito**: Implementar supervisión humana para decisiones legales críticas.

**Cubre**:
- Interrupts para aprobación
- Aprobación condicional
- Time travel (volver a estados anteriores)
- Patrones: cascada, paralelo
- Audit trail

**Cuándo usar**:
- Aprobación de demandas
- Revisión de contratos antes de enviar
- Decisiones con impacto legal significativo

**Ver más**: [`langgraph-human-in-the-loop-legal/SKILL.md`](./langgraph-human-in-the-loop-legal/SKILL.md)

---

## 🔧 Instalación de Skills Adicionales

Para agregar más skills del repositorio oficial:

```bash
# Todos los skills de LangChain
npx skills add langchain-ai/langchain-skills --skill langchain-fundamentals
npx skills add langchain-ai/langchain-skills --skill langchain-middleware
npx skills add langchain-ai/langchain-skills --skill langgraph-persistence

# Skills de Deep Agents
npx skills add langchain-ai/langchain-skills --skill deep-agents-memory
npx skills add langchain-ai/langchain-skills --skill deep-agents-orchestration
```

---

## 📊 Progreso de Implementación

| Categoría | Skills Oficiales | Skills Adaptados | Progreso |
|-----------|------------------|------------------|----------|
| LangChain | 3 | 1 | 33% |
| LangGraph | 3 | 2 | 67% |
| Deep Agents | 3 | 0 | 0% |
| **Total** | **11** | **3** | **27%** |

---

## 🎯 Próximos Pasos

1. [ ] Adaptar `langchain-fundamentals` para legal
2. [ ] Adaptar `langchain-middleware` para legal
3. [ ] Adaptar `langgraph-persistence` para legal
4. [ ] Adaptar `deep-agents-core` para legal
5. [ ] Adaptar `deep-agents-memory` para legal
6. [ ] Adaptar `deep-agents-orchestration` para legal

---

## 📚 Recursos Adicionales

### Documentación Oficial

- [LangChain Skills Repo](https://github.com/langchain-ai/langchain-skills)
- [Gemini CLI Skills Docs](https://geminicli.com/docs/cli/creating-skills/)
- [Agent Skills Spec](https://skills.sh/)

### Skills Relacionados

- [LangSmith Skills](https://github.com/langchain-ai/langsmith-skills) - Para observabilidad
- [LlamaIndex Skills](https://github.com/mindrally/skills) - Para RAG alternativo

---

## 🤝 Contribuciones

Para agregar nuevos skills:

1. Crear carpeta `agent_skills/nombre-del-skill/`
2. Agregar `SKILL.md` con YAML frontmatter
3. Agregar scripts, referencias y assets según necesite
4. Actualizar este README

---

*Skills creados: 2026-03-29*  
*Última actualización: 2026-03-29*  
*Basado en: [langchain-ai/langchain-skills](https://github.com/langchain-ai/langchain-skills)*
