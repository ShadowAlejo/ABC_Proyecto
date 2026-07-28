"""Clase concreta que envuelve el SVM (o Pipeline Scaler+PCA+LinearSVC) para clases faciales [REQ-FAC-04]."""
from typing import Any
import numpy as np
from classification.model_loader import load_svm_model

NUM_FACIAL_CLASSES = 16


class SVMFacialModel:
    """Envuelve el modelo SVM facial (bare SVC o Pipeline con PCA): expone prediccion y margen."""

    def __init__(self):
        self.model: Any = None       # puede ser SVC, LinearSVC, o Pipeline
        self.class_names: list[str] = []

    def load(self) -> None:
        bundle: dict[str, Any] = load_svm_model("models.svm_facial_path")
        self.model = bundle["model"]
        self.class_names = bundle.get(
            "class_names", [f"Sujeto_{i + 1}" for i in range(NUM_FACIAL_CLASSES)]
        )

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Devuelve (identidad_predicha, margen_de_decision_maximo).

        Compatible con:
          - Pipeline(StandardScaler, PCA, LinearSVC)  [nuevo]
          - SVC(kernel='linear')                      [legado]
        """
        if self.model is None:
            raise RuntimeError("El modelo SVM facial no ha sido cargado. Llame a load() primero.")

        features = feature_vector.reshape(1, -1)

        # Obtener el clasificador final (ultimo paso si es Pipeline, el modelo mismo si no)
        from sklearn.pipeline import Pipeline
        clf = self.model.steps[-1][1] if isinstance(self.model, Pipeline) else self.model

        # Transformar a traves del pipeline (scaler + PCA) antes de decision_function
        if isinstance(self.model, Pipeline):
            features_transformed = self.model[:-1].transform(features)
        else:
            features_transformed = features

        decision = clf.decision_function(features_transformed)[0]
        margin = float(np.max(decision)) if np.ndim(decision) > 0 else float(decision)

        pred_idx = int(self.model.predict(features)[0])
        identity = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)

        return identity, margin