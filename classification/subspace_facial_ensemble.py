"""Ensamble de Subespacios Faciales para SVM Facial [REQ-FAC-04].

Combina tres subespacios geométricos de características HOG faciales independientes (Score-Level Fusion):
1. Subespacio Global (0 a 1,200 dimensiones): Peso 40%
2. Subespacio Superior - Ojos/Cejas (1,200 a 2,880 dimensiones, 1,680 dims): Peso 35%
3. Subespacio Inferior - Nariz/Boca (2,880 a 3,840 dimensiones, 960 dims): Peso 25%

Utiliza Linear SVMs independientes con función de pérdida Modified Huber (suave y diferenciable),
proporcionando distribuciones de probabilidad calibradas independientes que se combinan mediante
suma ponderada: P_final = 0.40 * P_global + 0.35 * P_superior + 0.25 * P_inferior.
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

class SubspaceFacialEnsemble:
    """Ensamble de 3 SVMs independientes sobre subespacios geométricos HOG."""

    def __init__(self, alpha=1e-4, max_iter=1000, random_state=42):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler_global = StandardScaler()
        self.scaler_upper = StandardScaler()
        self.scaler_lower = StandardScaler()

        self.svm_global = SGDClassifier(loss="modified_huber", penalty="l2", alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
        self.svm_upper = SGDClassifier(loss="modified_huber", penalty="l2", alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
        self.svm_lower = SGDClassifier(loss="modified_huber", penalty="l2", alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)

        self.calibrated_global = None
        self.calibrated_upper = None
        self.calibrated_lower = None

        self.classes_ = None

    def partial_fit(self, X_tuple: tuple[np.ndarray, np.ndarray, np.ndarray], y: np.ndarray, classes: np.ndarray = None):
        """Ajuste incremental por mini-batches (Out-of-Core)."""
        X_g, X_u, X_l = X_tuple
        X_g = np.asarray(X_g, dtype=np.float32)
        X_u = np.asarray(X_u, dtype=np.float32)
        X_l = np.asarray(X_l, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        if classes is not None:
            self.classes_ = classes

        # 1. Partial fit scalers
        self.scaler_global.partial_fit(X_g)
        self.scaler_upper.partial_fit(X_u)
        self.scaler_lower.partial_fit(X_l)

        # 2. Transform (with partially fitted scalers so far)
        X_g_scaled = self.scaler_global.transform(X_g)
        X_u_scaled = self.scaler_upper.transform(X_u)
        X_l_scaled = self.scaler_lower.transform(X_l)

        # 3. Partial fit SVMs
        self.svm_global.partial_fit(X_g_scaled, y_arr, classes=self.classes_)
        self.svm_upper.partial_fit(X_u_scaled, y_arr, classes=self.classes_)
        self.svm_lower.partial_fit(X_l_scaled, y_arr, classes=self.classes_)

        return self

    def fit(self, X_tuple: tuple[np.ndarray, np.ndarray, np.ndarray], y: np.ndarray):
        """Ajusta en memoria usando partial_fit para compatibilidad."""
        return self.partial_fit(X_tuple, y, classes=np.unique(y))

    def calibrate(self, X_tuple: tuple[np.ndarray, np.ndarray, np.ndarray], y: np.ndarray):
        """Calibra las probabilidades reales (Platt Scaling) tras entrenar."""
        X_g, X_u, X_l = X_tuple
        X_g_scaled = self.scaler_global.transform(np.asarray(X_g, dtype=np.float32))
        X_u_scaled = self.scaler_upper.transform(np.asarray(X_u, dtype=np.float32))
        X_l_scaled = self.scaler_lower.transform(np.asarray(X_l, dtype=np.float32))
        y_arr = np.asarray(y, dtype=np.int64)

        self.calibrated_global = CalibratedClassifierCV(self.svm_global, cv="prefit", method="sigmoid")
        self.calibrated_global.fit(X_g_scaled, y_arr)

        self.calibrated_upper = CalibratedClassifierCV(self.svm_upper, cv="prefit", method="sigmoid")
        self.calibrated_upper.fit(X_u_scaled, y_arr)

        self.calibrated_lower = CalibratedClassifierCV(self.svm_lower, cv="prefit", method="sigmoid")
        self.calibrated_lower.fit(X_l_scaled, y_arr)

        return self

    def predict_proba(self, X_tuple: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Devuelve la distribución de probabilidad fusionada por pesos con Early-Stopping."""
        X_g, X_u, X_l = X_tuple
        X_g = np.asarray(X_g, dtype=np.float32)
        X_u = np.asarray(X_u, dtype=np.float32)
        X_l = np.asarray(X_l, dtype=np.float32)

        if X_g.ndim == 1:
            X_g = X_g.reshape(1, -1)
            X_u = X_u.reshape(1, -1)
            X_l = X_l.reshape(1, -1)

        # Evaluar primero la SVM Global (Early-Stopping)
        X_g_scaled = self.scaler_global.transform(X_g)
        if self.calibrated_global is not None:
            p_g = self.calibrated_global.predict_proba(X_g_scaled)
        else:
            p_g = self.svm_global.predict_proba(X_g_scaled)

        # Early-Stopping: Si la probabilidad Top-1 Global > 0.90, omitir SVMs regionales
        max_prob_g = np.max(p_g, axis=1)
        if np.all(max_prob_g > 0.90):
            return p_g

        # Evaluar SVMs regionales
        X_u_scaled = self.scaler_upper.transform(X_u)
        X_l_scaled = self.scaler_lower.transform(X_l)

        if self.calibrated_upper is not None:
            p_u = self.calibrated_upper.predict_proba(X_u_scaled)
            p_l = self.calibrated_lower.predict_proba(X_l_scaled)
        else:
            p_u = self.svm_upper.predict_proba(X_u_scaled)
            p_l = self.svm_lower.predict_proba(X_l_scaled)

        # Fusión ponderada: Global (40%), Superior (35%), Inferior (25%)
        return 0.40 * p_g + 0.35 * p_u + 0.25 * p_l

    def predict(self, X_tuple: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Devuelve el índice de clase con mayor probabilidad."""
        return np.argmax(self.predict_proba(X_tuple), axis=1)
