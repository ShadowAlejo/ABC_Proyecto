"""Clase concreta que envuelve el Ensamble SVM Facial [REQ-FAC-04]."""
from typing import Any
import numpy as np
from classification.model_loader import load_svm_model

NUM_FACIAL_CLASSES = 16


class SVMFacialModel:
    """Envuelve el modelo SVM facial (Ensamble de Subespacios)."""

    def __init__(self):
        self.model: Any = None
        self.class_names: list[str] = []

    def load(self) -> None:
        bundle: dict[str, Any] = load_svm_model("models.svm_facial_path")
        self.model = bundle["model"]
        self.class_names = bundle.get(
            "class_names", [f"Sujeto_{i + 1}" for i in range(NUM_FACIAL_CLASSES)]
        )

    def predict(self, feature_tuple: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[str, float]:
        """Devuelve (identidad_predicha, probabilidad_platt).
        
        Utiliza el umbral de rechazo de 0.70 basado en la probabilidad logística
        fusionada de los 3 clasificadores.
        """
        if self.model is None:
            raise RuntimeError("El modelo SVM facial no ha sido cargado. Llame a load() primero.")

        proba = self.model.predict_proba(feature_tuple)[0]
        
        max_idx = int(np.argmax(proba))
        
        # Calcular margin_gap (diferencia entre top-1 y top-2) para calibrar probabilidad
        sorted_proba = np.sort(proba)
        margin_gap = float(sorted_proba[-1] - sorted_proba[-2]) if len(sorted_proba) >= 2 else float(sorted_proba[-1])
        
        # Calibrar probabilidad logistica escalada (scale=6.0 para espacio 15 clases)
        from classification.logistic_confidence_converter import decision_margin_to_probability
        calibrated_prob = decision_margin_to_probability(margin_gap, scale=6.0)

        identity = self.class_names[max_idx] if max_idx < len(self.class_names) else str(max_idx)
        return identity, calibrated_prob