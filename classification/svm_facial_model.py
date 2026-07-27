"""Clase concreta que envuelve el SVM entrenado para 16 clases faciales [REQ-FAC-04]."""
from typing import Any
import numpy as np
from sklearn.svm import SVC
from classification.model_loader import load_svm_model
from utils.config_loader import ConfigLoader

NUM_FACIAL_CLASSES = 16


class SVMFacialModel:
    """Envuelve el modelo SVM facial: expone predicción y margen de decisión."""

    def __init__(self):
        self.model: SVC | None = None
        self.class_names: list[str] = []

    def load(self) -> None:
        bundle: dict[str, Any] = load_svm_model("models.svm_facial_path")
        self.model = bundle["model"]
        self.class_names = bundle.get(
            "class_names", [f"Sujeto_{i + 1}" for i in range(NUM_FACIAL_CLASSES)]
        )

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Devuelve (identidad_predicha, margen_de_decision_maximo)."""
        if self.model is None:
            raise RuntimeError("El modelo SVM facial no ha sido cargado. Llame a load() primero.")

        features = feature_vector.reshape(1, -1)
        decision = self.model.decision_function(features)[0]
        margin = float(np.max(decision)) if np.ndim(decision) > 0 else float(decision)

        pred_idx = int(self.model.predict(features)[0])
        identity = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)

        return identity, margin

    def fit(self, X: np.ndarray, y: np.ndarray, class_weight: str | dict = "balanced", C: float = 1.0) -> None:
        """Entrena el modelo con ponderación de clases [REQ-ENT-01]."""
        self.model = SVC(kernel="linear", C=C, class_weight=class_weight, probability=False)
        self.model.fit(X, y)