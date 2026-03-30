"""
Módulo Human in the Loop - Aprobación Humana

Implementa mecanismos para solicitar aprobación humana antes de enviar
respuestas críticas o de baja confianza.

Author: Curso LangChain + LangGraph para RAG
Version: 1.0.0
"""

import logging
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum

from .config import (
    HITL_ENABLED,
    HITL_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ApprovalDecision(Enum):
    """Decisiones posibles de aprobación humana."""
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    SKIPPED = "skipped"


class HumanApproval:
    """
    Sistema de aprobación humana para respuestas del chatbot.

    Características:
    - Evaluación automática de temas críticos
    - Umbrales de confianza para aprobación automática
    - Registro de auditoría de decisiones

    Ejemplo:
        >>> hitl = HumanApproval()
        >>> approved, feedback = hitl.request_approval(
        ...     response="La tutela procede en...",
        ...     context="Usuario pregunta sobre plazos"
        ... )
    """

    # Temas críticos que siempre requieren aprobación humana
    CRITICAL_TOPICS = [
        "demanda",
        "juicio",
        "sentencia",
        "cárcel",
        "detención",
        "multa",
        "sanción",
        "pérdida",
        "custodia",
        "divorcio",
        "herencia",
        "testamento",
        "quiebra",
        "embargo",
        "desalojo",
        "despido",
        "accidente",
        "muerte",
        "lesiones",
        "tutela",  # Aunque es básico, puede tener implicaciones graves
    ]

    # Palabras que indican urgencia/alto riesgo
    URGENCY_INDICATORS = [
        "urgente",
        "emergencia",
        "inmediato",
        "ya mismo",
        "rápido",
        "prioritario",
        "crítico",
        "grave",
        "peligro",
        "amenaza",
    ]

    def __init__(
        self,
        enabled: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
        auto_approve_non_critical: bool = True,
    ):
        """
        Inicializar el sistema de aprobación humana.

        Args:
            enabled: Si habilitar HITL
            confidence_threshold: Umbral de confianza para aprobación automática
            auto_approve_non_critical: Aprobar automáticamente temas no críticos
        """
        self.enabled = enabled if enabled is not None else HITL_ENABLED
        self.confidence_threshold = confidence_threshold or HITL_CONFIDENCE_THRESHOLD
        self.auto_approve_non_critical = auto_approve_non_critical

        # Registro de auditoría
        self._audit_log: List[Dict[str, Any]] = []

        logger.debug(
            f"HumanApproval inicializado: enabled={self.enabled}, "
            f"threshold={self.confidence_threshold}"
        )

    def check_critical_topics(self, query: str) -> bool:
        """
        Verificar si la consulta toca temas críticos.

        Args:
            query: Consulta del usuario

        Returns:
            True si es tema crítico, False si no
        """
        query_lower = query.lower()

        # Verificar temas críticos
        for topic in self.CRITICAL_TOPICS:
            if topic in query_lower:
                logger.debug(f"Tema crítico detectado: '{topic}'")
                return True

        # Verificar indicadores de urgencia
        for indicator in self.URGENCY_INDICATORS:
            if indicator in query_lower:
                logger.debug(f"Indicador de urgencia detectado: '{indicator}'")
                return True

        return False

    def should_require_approval(
        self,
        response: str,
        confidence: float,
        is_critical_topic: bool = False,
    ) -> bool:
        """
        Decidir si se requiere aprobación humana.

        Args:
            response: Respuesta generada
            confidence: Confianza estimada de la respuesta
            is_critical_topic: Si el tema es crítico

        Returns:
            True si requiere aprobación, False si puede proceder
        """
        # Si HITL está deshabilitado, nunca requiere aprobación
        if not self.enabled:
            return False

        # Temas críticos siempre requieren aprobación
        if is_critical_topic:
            logger.info("Aprobación requerida: tema crítico")
            return True

        # Baja confianza requiere aprobación
        if confidence < self.confidence_threshold:
            logger.info(f"Aprobación requerida: confianza baja ({confidence:.2f})")
            return True

        # Si está configurado para aprobar automáticamente no críticos
        if self.auto_approve_non_critical:
            logger.info("Aprobación omitida: tema no crítico, confianza adecuada")
            return False

        # Por defecto, requerir aprobación para todo
        return True

    def request_approval(
        self,
        response: str,
        context: str = "",
        auto_approve: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Solicitar aprobación humana para una respuesta.

        En modo CLI, espera input del usuario.
        En modo automático, puede aprobar/rechazar basándose en reglas.

        Args:
            response: Respuesta a aprobar
            context: Contexto de la conversación
            auto_approve: Si aprobar automáticamente (saltar validación)

        Returns:
            Tuple de (aprobado, feedback)
            - aprobado: True si se aprobó, False si se rechazó
            - feedback: Respuesta editada o razón del rechazo
        """
        # Si auto_approve está activado, aprobar directamente
        if auto_approve:
            logger.debug("Aprobación automática activada")
            self._log_decision(ApprovalDecision.SKIPPED, response, "Auto-aprobado")
            return True, None

        # Si HITL está deshabilitado, aprobar automáticamente
        if not self.enabled:
            logger.debug("HITL deshabilitado, aprobación automática")
            self._log_decision(ApprovalDecision.SKIPPED, response, "HITL deshabilitado")
            return True, None

        # Modo interactivo CLI
        print("\n" + "=" * 80)
        print("⚠️  REVISIÓN HUMANA REQUERIDA")
        print("=" * 80)
        print("\n📝 Respuesta generada:")
        print("-" * 80)
        print(response)
        print("-" * 80)

        if context:
            print("\n📌 Contexto:")
            print(f"{context[:500]}..." if len(context) > 500 else context)

        print("\n\nOpciones:")
        print("  [A] Aprobar respuesta")
        print("  [E] Editar respuesta")
        print("  [R] Rechazar respuesta")
        print("  [S] Saltar revisión (aprobar automáticamente)")
        print()

        while True:
            try:
                choice = input("Tu decisión [A/E/R/S]: ").strip().lower()

                if choice in ["a", "aprobar", ""]:
                    logger.info("Respuesta aprobada por humano")
                    self._log_decision(ApprovalDecision.APPROVED, response)
                    return True, None

                elif choice in ["e", "editar", "edit"]:
                    print("\n✏️  Edita la respuesta (deja vacío para cancelar):")
                    edited_response = input("> ").strip()

                    if edited_response:
                        logger.info("Respuesta editada por humano")
                        self._log_decision(ApprovalDecision.EDITED, edited_response)
                        return True, edited_response
                    else:
                        print("Edición cancelada")
                        continue

                elif choice in ["r", "rechazar", "reject"]:
                    print("\n❌ Razón del rechazo (opcional):")
                    reason = input("> ").strip()

                    logger.info(f"Respuesta rechazada por humano: {reason or 'Sin razón'}")
                    self._log_decision(ApprovalDecision.REJECTED, response, reason)
                    return False, reason or "Rechazada por revisor humano"

                elif choice in ["s", "saltar", "skip"]:
                    logger.info("Revisión saltada por usuario")
                    self._log_decision(ApprovalDecision.SKIPPED, response)
                    return True, "Saltado por usuario"

                else:
                    print("Opción no válida. Por favor ingresa A, E, R o S")

            except KeyboardInterrupt:
                logger.info("Aprobación interrumpida por usuario")
                return False, "Interrumpido por usuario"

            except Exception as e:
                logger.error(f"Error en solicitud de aprobación: {e}")
                return False, "Error en proceso de aprobación"

    def _log_decision(
        self,
        decision: ApprovalDecision,
        response: str,
        reason: str = "",
    ):
        """
        Registrar decisión en el log de auditoría.

        Args:
            decision: Decisión tomada
            response: Respuesta involucrada
            reason: Razón o feedback adicional
        """
        from datetime import datetime

        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision.value,
            "response_preview": response[:100],
            "reason": reason,
        })

        # Mantener solo últimos 100 registros
        if len(self._audit_log) > 100:
            self._audit_log = self._audit_log[-100:]

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Obtener el registro de auditoría.

        Returns:
            Lista de registros de decisiones
        """
        return self._audit_log.copy()

    def clear_audit_log(self):
        """Limpiar el registro de auditoría."""
        self._audit_log.clear()
        logger.debug("Audit log limpiado")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de aprobaciones.

        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_decisions": len(self._audit_log),
            "approved": 0,
            "rejected": 0,
            "edited": 0,
            "skipped": 0,
        }

        for record in self._audit_log:
            decision = record["decision"]
            if decision in stats:
                stats[decision] += 1

        return stats

    def __repr__(self) -> str:
        return f"HumanApproval(enabled={self.enabled}, threshold={self.confidence_threshold}, decisions={len(self._audit_log)})"


# Funciones de conveniencia
def quick_approval_check(
    response: str,
    confidence: float,
    query: str = "",
) -> bool:
    """
    Verificación rápida de si requiere aprobación.

    Args:
        response: Respuesta generada
        confidence: Confianza estimada
        query: Consulta original (para detectar temas críticos)

    Returns:
        True si requiere aprobación
    """
    hitl = HumanApproval()
    is_critical = hitl.check_critical_topics(query) if query else False
    return hitl.should_require_approval(response, confidence, is_critical)
