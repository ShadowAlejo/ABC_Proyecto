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

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Devuelve (identidad_predicha, probabilidad_platt).
        
        Utiliza el umbral de rechazo de 0.70 basado en la probabilidad logística
        fusionada de los 3 clasificadores.
        """
        if self.model is None:
            raise RuntimeError("El modelo SVM facial no ha sido cargado. Llame a load() primero.")

        features = feature_vector.reshape(1, -1)

        # Ensamble devuelve probabilidades Platt de LogisticRegression
        proba = self.model.predict_proba(features)[0]
        
        max_idx = int(np.argmax(proba))
        max_prob = float(proba[max_idx])

        # NOTA: Se eliminó el umbral interno estricto. Ahora el modelo siempre 
        # devuelve la clase con mayor probabilidad. La responsabilidad de rechazar
        # (Desconocido) recae íntegramente en WeightedVotingInertia y el Gate (0.65)
        # del PipelineOrchestrator, quienes agregan esta señal ruidosa a lo largo del tiempo.

        identity = self.class_names[max_idx] if max_idx < len(self.class_names) else str(max_idx)
        return identity, max_prob