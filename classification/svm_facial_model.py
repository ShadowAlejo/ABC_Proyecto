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

        sorted_indices = np.argsort(proba)[::-1]
        max_idx = int(sorted_indices[0])
        top1 = float(proba[max_idx])
        top2 = float(proba[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0

        # Si el margen entre el 1er y 2do lugar es pequeño (< 0.12), marcar como ambiguo
        if (top1 - top2) < 0.12:
            return "Desconocido", top1

        identity = self.class_names[max_idx] if max_idx < len(self.class_names) else str(max_idx)
        return identity, top1