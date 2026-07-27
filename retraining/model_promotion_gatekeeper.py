"""Compara métricas del modelo nuevo vs producción y decide si reemplazarlo [REQ-ENT-05]."""
from dataclasses import dataclass
from retraining.cross_validation_runner import CrossValidationResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PromotionDecision:
    promote: bool
    reason: str


class ModelPromotionGatekeeper:
    def __init__(self, min_improvement: float = 0.01, min_absolute_f1: float = 0.75):
        """
        min_improvement: mejora mínima requerida en F1-macro sobre el modelo en producción.
        min_absolute_f1: umbral mínimo absoluto de calidad para evitar sobreajuste a una sola sesión.
        """
        self.min_improvement = min_improvement
        self.min_absolute_f1 = min_absolute_f1

    def evaluate_promotion(self, new_result: CrossValidationResult,
                            production_result: CrossValidationResult | None) -> PromotionDecision:
        if new_result.mean_f1_macro < self.min_absolute_f1:
            return PromotionDecision(
                promote=False,
                reason=f"F1-macro del nuevo modelo ({new_result.mean_f1_macro:.3f}) "
                       f"por debajo del umbral mínimo ({self.min_absolute_f1})."
            )

        if production_result is None:
            return PromotionDecision(promote=True, reason="No existe modelo en producción; se promueve por defecto.")

        improvement = new_result.mean_f1_macro - production_result.mean_f1_macro
        if improvement >= self.min_improvement:
            return PromotionDecision(
                promote=True,
                reason=f"Mejora de F1-macro de {improvement:.3f} >= umbral requerido ({self.min_improvement})."
            )

        return PromotionDecision(
            promote=False,
            reason=f"Mejora insuficiente ({improvement:.3f} < {self.min_improvement}); "
                   f"posible sobreajuste a la sesión actual."
        )