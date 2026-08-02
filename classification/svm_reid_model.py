"""Clase concreta equivalente para el modelo SVM de Re-ID basado en vectores LBP [REQ-RID-06]."""
from typing import Any
import numpy as np
from classification.model_loader import load_svm_model


class SVMReidModel:
    """Envuelve el modelo Re-ID calibrado: expone predicción y probabilidad directa."""

    def __init__(self):
        self.model = None
        self.class_names: list[str] = []

    def load(self) -> None:
        bundle: dict[str, Any] = load_svm_model("models.svm_reid_path")
        self.model = bundle["model"]
        self.class_names = bundle.get("class_names", [])

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("El modelo SVM Re-ID no ha sido cargado. Llame a load() primero.")

        # El modelo ahora es un Pipeline: StandardScaler -> CalibratedClassifierCV
        vector = feature_vector.reshape(1, -1)
        probas = self.model.predict_proba(vector)[0]
        
        pred_idx = int(np.argmax(probas))
        probability = float(probas[pred_idx])

        identity = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)

        return identity, probability