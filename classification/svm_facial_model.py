"""Clase concreta que envuelve el SVM (o Pipeline Scaler+PCA+LinearSVC) para clases faciales [REQ-FAC-04]."""
from typing import Any
import numpy as np
from classification.model_loader import load_svm_model

NUM_FACIAL_CLASSES = 16


class SVMFacialModel:
    """Envuelve el modelo SVM facial (bare SVC o Pipeline con PCA): expone prediccion y margen.

    La confianza retornada es el 'margin_gap': la diferencia entre el margen de la clase
    ganadora y el margen de la segunda clase. Esto es mucho mas discriminativo que usar
    solo el margen maximo:
      - Persona conocida clara: gap alto (ganador domina) -> alta confianza
      - Persona desconocida: gap bajo (varias clases compiten) -> baja confianza
    """

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
        """Devuelve (identidad_predicha, margin_gap).

        margin_gap = max(decision) - second_max(decision):
          - Si hay solo 1 clase: usa max(decision) como fallback.
          - Si hay >= 2 clases: usa la brecha entre el 1er y 2do clasificador OVR.

        Compatible con:
          - Pipeline(StandardScaler, PCA, LinearSVC)  [nuevo]
          - SVC(kernel='linear')                      [legado]
        """
        if self.model is None:
            raise RuntimeError("El modelo SVM facial no ha sido cargado. Llame a load() primero.")

        features = feature_vector.reshape(1, -1)

        # -- Una sola pasada: transformar con scaler+PCA, luego usar el clasificador final --
        from sklearn.pipeline import Pipeline
        is_pipeline = isinstance(self.model, Pipeline)
        clf = self.model.steps[-1][1] if is_pipeline else self.model

        # Transformar UNA sola vez (evita doble pasada costosa)
        features_t = self.model[:-1].transform(features) if is_pipeline else features

        # Vector de decision OVR (un margen por clase)
        decision = clf.decision_function(features_t)[0]

        # -- Margin gap: brecha ganador vs segundo lugar (mas discriminativo para desconocidos) --
        if np.ndim(decision) > 0 and len(decision) >= 2:
            top2 = np.partition(decision, -2)[-2:]   # los dos margenes mas altos
            margin_gap = float(top2[-1] - top2[-2])  # max - second_max
        else:
            # Caso binario o 1 clase: usar el margen directamente
            margin_gap = float(decision) if np.ndim(decision) == 0 else float(np.max(decision))

        # Predecir la clase ganadora directamente desde los datos ya transformados
        pred_idx = int(clf.predict(features_t)[0])
        identity = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)

        return identity, margin_gap