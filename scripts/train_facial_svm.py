"""Entrena el Ensamble SVM Facial (Global, Ojos, Nariz/Boca) con Opponent-HOG.

Optimizaciones aplicadas:
  - Jittering espacial (+-2 px) para robustez.
  - Reduccion de dimensionalidad: PCA(whiten) + LDA + L2 Normalization.
  - Ensamble de subespacios combinado por regresion logistica (Platt Scaling).
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classification.svm_facial_model import SVMFacialModel
from feature_extraction.face.yunet_face_detector import YuNetFaceDetector
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from feature_extraction.face.face_quality_validator import validate_face_quality
from retraining.class_balancer import compute_class_weights
from retraining.data_augmentation_engine import DataAugmentationEngine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score as sk_f1
from utils.file_io_helpers import list_files, save_pickle, load_pickle
from utils.logger import get_logger

logger = get_logger("train_facial_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_facial/svm_facial_model.pkl")
MIN_SAMPLES_FOR_AUGMENTATION = 30
MAX_VECTORS_PER_CLASS = 700

_SCALE_VARIANTS = [0.65, 0.80, 1.25]
_ROTATION_VARIANTS = [-7, 7]
_JITTER_VARIANTS = [(-2, 0), (2, 0), (0, -2), (0, 2)]


def _extract_vector(img: np.ndarray, face_result) -> np.ndarray | None:
    face_bgr, landmarks = normalize_face(img, face_result)
    if face_bgr is None: return None
    return extract_hog_features(face_bgr, landmarks)


def _jitter_image(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def _extract_from_synthetic(img: np.ndarray, detector: YuNetFaceDetector) -> np.ndarray | None:
    face_result = detector.detect_training(img)
    if not face_result.detected: return None
    return _extract_vector(img, face_result)


def build_dataset(face_detector: YuNetFaceDetector, augmenter: DataAugmentationEngine):
    class_names = sorted([d.name for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise RuntimeError("No hay clases.")
    
    X, y = [], []
    for class_idx, class_name in enumerate(class_names):
        image_paths = list_files(RAW_IMAGES_DIR / class_name, (".jpg", ".jpeg", ".png"))
        class_vectors = []
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            face_result = face_detector.detect_training(img)
            if not face_result.detected: continue
            
            vector = _extract_vector(img, face_result)
            if vector is not None:
                class_vectors.append(vector)
                
                # Spatial Jittering
                for tx, ty in _JITTER_VARIANTS:
                    jit_img = _jitter_image(img, tx, ty)
                    jit_vec = _extract_from_synthetic(jit_img, face_detector)
                    if jit_vec is not None:
                        class_vectors.append(jit_vec)
                        
                # Scale & Rotation
                for scale in _SCALE_VARIANTS:
                    scale_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
                    sv = _extract_from_synthetic(scale_img, face_detector)
                    if sv is not None: class_vectors.append(sv)
        
        for v in class_vectors[:MAX_VECTORS_PER_CLASS]:
            X.append(v)
            y.append(class_idx)
            
        logger.info(f"[{class_name}] -> {len(class_vectors)} vectores")
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), class_names


class SubspaceEnsembleSVM:
    """Ensamble de Pipelines (PCA+LDA+L2+LogisticRegression) sobre sub-porciones de HOG."""
    def __init__(self, n_classes, n_samples):
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.pipelines = {}
        self.weights = {"global": 0.5, "eyes": 0.3, "mouth": 0.2}

    def _build_pipe(self):
        n_pca = min(500, self.n_samples - 1)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, whiten=True, random_state=42)),
            ("lda", LinearDiscriminantAnalysis()),
            ("l2", Normalizer(norm='l2')),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, multi_class='multinomial', random_state=42))
        ])

    def _slice_features(self, X):
        # 16176 features per channel: 13584 global + 2592 local
        # Total dims = 48528
        # We simplify slicing by doing:
        # Global: all channels global part (13584 * 3)
        # Eyes: patches 0, 1, 5 from all channels
        # Mouth: patches 2, 3, 4 from all channels
        
        X_global, X_eyes, X_mouth = [], [], []
        for row in X:
            r_g, r_e, r_m = [], [], []
            for ch in range(3):
                offset = ch * 16176
                r_g.extend(row[offset : offset + 13584])
                
                local_off = offset + 13584
                # Eyes: 0, 1, 5
                r_e.extend(row[local_off + 0*432 : local_off + 2*432])
                r_e.extend(row[local_off + 5*432 : local_off + 6*432])
                # Mouth/Nose: 2, 3, 4
                r_m.extend(row[local_off + 2*432 : local_off + 5*432])
                
            X_global.append(r_g)
            X_eyes.append(r_e)
            X_mouth.append(r_m)
            
        return np.array(X_global), np.array(X_eyes), np.array(X_mouth)

    def fit(self, X, y):
        X_g, X_e, X_m = self._slice_features(X)
        self.pipelines["global"] = self._build_pipe().fit(X_g, y)
        self.pipelines["eyes"] = self._build_pipe().fit(X_e, y)
        self.pipelines["mouth"] = self._build_pipe().fit(X_m, y)
        return self

    def predict_proba(self, X):
        X_g, X_e, X_m = self._slice_features(X)
        p_g = self.pipelines["global"].predict_proba(X_g)
        p_e = self.pipelines["eyes"].predict_proba(X_e)
        p_m = self.pipelines["mouth"].predict_proba(X_m)
        
        return (p_g * self.weights["global"] + 
                p_e * self.weights["eyes"] + 
                p_m * self.weights["mouth"])

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def main():
    face_detector = YuNetFaceDetector()
    augmenter = DataAugmentationEngine()

    X, y, class_names = build_dataset(face_detector, augmenter)
    logger.info(f"Dataset construido: {X.shape[0]} muestras x {X.shape[1]} dims")

    if len(class_names) < 2:
        logger.error("Se necesitan al menos 2 clases para LDA.")
        return

    # Usar un wrapper compatible con sklearn cross_validate
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class EnsembleWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, n_c, n_s):
            self.model = SubspaceEnsembleSVM(n_c, n_s)
            self.classes_ = np.arange(n_c)
        def fit(self, X, y):
            self.model.fit(X, y)
            return self
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        def predict(self, X):
            return self.model.predict(X)

    wrapper = EnsembleWrapper(len(class_names), X.shape[0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_results = cross_validate(
            wrapper, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring={"f1_macro": make_scorer(sk_f1, average="macro", zero_division=0)},
            n_jobs=1 # Para no saturar RAM con las matrices grandes
        )
        
    mean_f1 = np.mean(cv_results["test_f1_macro"])
    logger.info(f"F1-Macro CV: {mean_f1:.4f}")

    if mean_f1 >= 0.30:
        logger.info("Entrenando final...")
        wrapper.fit(X, y)
        OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_pickle({
            "model": wrapper.model,
            "class_names": class_names,
            "cv_f1_macro": mean_f1
        }, OUTPUT_MODEL_PATH)
        logger.info("[OK] Guardado.")


if __name__ == "__main__":
    main()
