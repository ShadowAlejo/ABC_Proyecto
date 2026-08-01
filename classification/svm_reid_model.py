"""Clase concreta equivalente para el modelo SVM de Re-ID basado en vectores LBP [REQ-RID-06]."""
from typing import Any
import numpy as np
from sklearn.svm import SVC
from classification.model_loader import load_svm_model


class SVMReidModel:
    """Envuelve el modelo SVM de Re-ID: expone predicción y margen de decisión."""

    def __init__(self):
        self.model: SVC | None = None
        self.class_names: list[str] = []

    def load(self) -> None:
        bundle: dict[str, Any] = load_svm_model("models.svm_reid_path")
        self.model = bundle["model"]
        self.class_names = bundle.get("class_names", [])

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("El modelo SVM Re-ID no ha sido cargado. Llame a load() primero.")

        features = feature_vector.reshape(1, -1)
        decision = self.model.decision_function(features)[0]
        
        # Calcular margin_gap (max - second_max) para el convertidor logístico
        if len(decision) >= 2:
            sorted_dec = np.sort(decision)
            margin_gap = float(sorted_dec[-1] - sorted_dec[-2])
        else:
            margin_gap = float(decision) if np.ndim(decision) == 0 else float(decision[0])

        pred_idx = int(self.model.predict(features)[0])
        identity = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)

        return identity, margin_gap

    def fit(self, X: np.ndarray, y: np.ndarray, class_weight: str | dict = "balanced", C: float = 1.0) -> None:
        self.model = SVC(kernel="linear", C=C, class_weight=class_weight, probability=False)
        self.model.fit(X, y)