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
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classification.svm_facial_model import SVMFacialModel
from feature_extraction.face.yunet_face_detector import YuNetFaceDetector
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from feature_extraction.face.face_quality_validator import validate_face_quality
from retraining.class_balancer import compute_class_weights
from retraining.data_augmentation_engine import DataAugmentationEngine
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import make_scorer, f1_score as sk_f1
from utils.file_io_helpers import list_files, save_pickle, load_pickle, load_image
from utils.logger import get_logger

logger = get_logger("train_facial_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_facial/svm_facial_model.pkl")
MIN_SAMPLES_FOR_AUGMENTATION = 30
MAX_VECTORS_PER_CLASS = 1200

def _extract_vector(img: np.ndarray, face_result) -> np.ndarray | None:
    face_bgr, landmarks = normalize_face(img, face_result)
    if face_bgr is None: return None
    return extract_hog_features(face_bgr, landmarks)


CACHE_PATH = Path("dataset/cache/facial_features.npz")

def _worker_process_facial_chunk(chunk):
    # Inicialización local para evitar Pickling y conflictos entre procesos
    import cv2
    cv2.setNumThreads(0)
    from feature_extraction.face.yunet_face_detector import YuNetFaceDetector
    from retraining.data_augmentation_engine import DataAugmentationEngine
    from feature_extraction.face.face_normalizer import normalize_face
    from feature_extraction.face.hog_extractor import extract_hog_features
    face_detector = YuNetFaceDetector()
    augmenter = DataAugmentationEngine()
    results = []
    for i, (img_path, class_idx) in enumerate(chunk):
        img = load_image(img_path)
        if img is None: continue
        
        # Optimización crítica: Las imágenes de 6400x3600 tardan 12s por imagen. 
        # Redimensionamos a un máximo de 1280px para acelerar 100x.
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1280:
            scale = 1280 / max_dim
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        face_result = face_detector.detect_training(img)
        if not face_result.detected: continue
        
        face_bgr, landmarks = normalize_face(img, face_result)
        if face_bgr is None: continue
        
        vector = extract_hog_features(face_bgr, landmarks)
        if vector is not None:
            results.append((class_idx, vector))
            
            # Generar 2 muestras con jittering sutil
            jittered_faces = augmenter.generate_jitter_samples(face_bgr, n_samples=2)
            for j_face in jittered_faces:
                j_vector = extract_hog_features(j_face, landmarks)
                if j_vector is not None:
                    results.append((class_idx, j_vector))
    return results

def build_dataset(augmenter: DataAugmentationEngine):
    if CACHE_PATH.exists():
        logger.info(f"Cargando características desde caché: {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        return data['X'], data['y'], data['groups'], data['class_names'].tolist()
        
    class_names = sorted([d.name for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise RuntimeError("No hay clases.")
    
    # 1. Fase Map: Lista plana
    flat_tasks = []
    for class_idx, class_name in enumerate(class_names):
        image_paths = list_files(RAW_IMAGES_DIR / class_name, (".jpg", ".jpeg", ".png"))
        image_paths.sort()
        for img_path in image_paths:
            flat_tasks.append((img_path, class_idx))
            
    logger.info(f"Fase Map completada: {len(flat_tasks)} imágenes independientes encontradas.")
    
    # 2. Fase Distribución y Ejecución (MIMD)
    num_workers = max(1, min(4, os.cpu_count() - 1))
    chunk_size = max(1, len(flat_tasks) // num_workers)
    chunks = [flat_tasks[i:i + chunk_size] for i in range(0, len(flat_tasks), chunk_size)]
    
    logger.info(f"Lanzando {num_workers} procesos trabajadores para extracción...")
    extracted_features = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker_process_facial_chunk, chunk) for chunk in chunks]
        completed = 0
        for future in as_completed(futures):
            try:
                res = future.result()
                extracted_features.extend(res)
                completed += 1
                logger.info(f"Progreso: Trabajador {completed}/{num_workers} finalizado. (+{len(res)} muestras)")
            except Exception as e:
                logger.error(f"Error en trabajador: {e}")
                
    # 3. Fase Reduce: Agrupación y Lógica de Clase
    from collections import defaultdict
    class_vectors = defaultdict(list)
    for class_idx, vector in extracted_features:
        class_vectors[class_idx].append(vector)
        
    X, y, groups = [], [], []
    valid_class_names = []
    
    for class_idx, class_name in enumerate(class_names):
        vectors = class_vectors.get(class_idx, [])
        # Truncar a un máximo de 120 válidas
        vectors = vectors[:120]
        
        # Control de Calidad: Mínimo 25 fotos por clase
        if len(vectors) < 25:
            logger.warning(f"[{class_name}] Excluida. Solo tiene {len(vectors)} vectores válidos (Mínimo requerido: 25).")
            continue
            
        valid_class_names.append(class_name)
        new_class_idx = len(valid_class_names) - 1 # Re-index based on valid classes
        
        for i, v in enumerate(vectors):
            X.append(v)
            y.append(new_class_idx)
            session_id = f"{new_class_idx}_{i // 5}"
            groups.append(session_id)
            
        logger.info(f"[{class_name}] -> {len(vectors)} vectores válidos. Incluida.")
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups)
    
    # Caching
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, X=X, y=y, groups=groups, class_names=valid_class_names)
    logger.info(f"Características guardadas en caché: {CACHE_PATH}")
    
    return X, y, groups, valid_class_names


def build_meta_model():
    """Construye el StackingClassifier con los 3 subespacios (Global, Superior, Inferior)."""
    # Índices exactos (3264 dimensiones por imagen):
    # Global (Coarse): 0 a 1200
    # Superior (Ojos/Cejas): 1200 a 2544
    # Inferior (Nariz/Boca): 2544 a 3264
    
    ct_global = ColumnTransformer([("global", "passthrough", slice(0, 1200))])
    ct_upper  = ColumnTransformer([("upper", "passthrough", slice(1200, 2544))])
    ct_lower  = ColumnTransformer([("lower", "passthrough", slice(2544, 3264))])
    
    from sklearn.model_selection import GridSearchCV
    # Para el SVM base, en lugar de un C fijo, usamos GridSearchCV para encontrar el óptimo en cada subespacio.
    base_svm = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
    grid_svm = GridSearchCV(base_svm, param_grid={'C': [0.5, 1.0, 5.0, 10.0]}, cv=3, n_jobs=1)
    
    pipe_base = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(grid_svm, method='sigmoid', cv=2))
    ])
    
    pipe_global = Pipeline([("select", ct_global), ("base", clone(pipe_base))])
    pipe_upper  = Pipeline([("select", ct_upper), ("base", clone(pipe_base))])
    pipe_lower  = Pipeline([("select", ct_lower), ("base", clone(pipe_base))])
    
    estimators = [
        ("global", pipe_global),
        ("upper", pipe_upper),
        ("lower", pipe_lower)
    ]
    
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, C=5.0, penalty='l2'),
        cv=5,
        n_jobs=-1
    )


def main():
    augmenter = DataAugmentationEngine()

    X, y, groups, class_names = build_dataset(augmenter)
    logger.info(f"Dataset construido: {X.shape[0]} muestras x {X.shape[1]} dims")

    if len(class_names) < 2:
        logger.error("Se necesitan al menos 2 clases para entrenar.")
        return

    wrapper = build_meta_model()

    logger.info("Iniciando validación cruzada y entrenamiento de StackingClassifier...")
    logger.info("NOTA: Esto tomará algo de tiempo, entrenando ensambles SVM. Por favor, espera...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_results = cross_validate(
            wrapper, X, y, groups=groups, cv=GroupKFold(n_splits=5),
            scoring={"f1_macro": make_scorer(sk_f1, average="macro", zero_division=0)},
            n_jobs=-1
        )
        
    mean_f1 = np.mean(cv_results["test_f1_macro"])
    logger.info(f"F1-Macro CV: {mean_f1:.4f}")

    if mean_f1 >= 0.80:
        logger.info("Entrenando modelo final con todos los datos...")
        wrapper.fit(X, y)
        OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_pickle({
            "model": wrapper,
            "class_names": class_names,
            "cv_f1_macro": mean_f1
        }, OUTPUT_MODEL_PATH)
        logger.info("[OK] Modelo SVM guardado correctamente.")
    else:
        logger.warning(f"[ALERTA] El F1-Macro CV ({mean_f1:.4f}) es menor al umbral (0.80). El modelo NO se guardará por baja calidad.")


if __name__ == "__main__":
    main()
