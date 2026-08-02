"""Entrena el SVM Re-ID (1,888 dimensiones LBP Hellinger) sobre la silueta corporal completa.

Arquitectura:
  - Carga imágenes desde `dataset/captures` (75 imágenes base por clase).
  - Aumentación geométrica/estructural (Flip, Escala, Traslación, Cutout) -> 450 vectores/clase.
  - Extracción de LBP-U (R=1, P=8) en rejilla 4x8 con normalización Hellinger (L1-sqrt).
  - Entrenamiento directo en memoria RAM de `StandardScaler` + `CalibratedClassifierCV(LinearSVC)`.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_extraction.body.body_roi_isolator import isolate_body_roi
from feature_extraction.body.spatial_grid_histogram import extract_spatial_grid_lbp
from retraining.data_augmentation_engine import DataAugmentationEngine
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils.file_io_helpers import list_files, save_pickle
from utils.logger import get_logger

logger = get_logger("train_reid_svm")

CAPTURES_DIR = Path("dataset/captures")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_reid/svm_reid_model.pkl")

def _extract_lbp(img: np.ndarray) -> np.ndarray | None:
    """Aísla cuerpo completo y extrae 1,888 dimensiones LBP."""
    try:
        body_128x256 = isolate_body_roi(img)
        gray = cv2.cvtColor(body_128x256, cv2.COLOR_BGR2GRAY)
        return extract_spatial_grid_lbp(gray)
    except Exception as e:
        logger.debug(f"Error LBP: {e}")
        return None

def main():
    logger.info("Iniciando entrenamiento definitivo Re-ID (1,888 dims)...")
    
    person_dirs = [d for d in CAPTURES_DIR.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in person_dirs])
    
    if len(class_names) < 2:
        logger.error(f"Se necesitan al menos 2 clases en {CAPTURES_DIR}")
        return

    augmenter = DataAugmentationEngine()
    X_list = []
    y_list = []

    logger.info("Extrayendo características LBP y aplicando aumentación geométrica en RAM...")
    
    for class_idx, class_name in enumerate(class_names):
        p_dir = CAPTURES_DIR / class_name
        files = list_files(p_dir, extensions=[".jpg", ".jpeg", ".png"])
        files.sort()
        
        # Limitar a las 75 imágenes base reportadas (o máximo disponible)
        selected_files = files[:75]
        
        valid_count = 0
        for f in selected_files:
            img = cv2.imread(str(f))
            if img is None:
                continue
                
            # Vector original
            vec = _extract_lbp(img)
            if vec is not None:
                X_list.append(vec)
                y_list.append(class_idx)
                valid_count += 1
                
            # 5 Variantes geométricas
            variants = augmenter.generate_reid_geometric_samples(img, n_samples=5)
            for v_img in variants:
                v_vec = _extract_lbp(v_img)
                if v_vec is not None:
                    X_list.append(v_vec)
                    y_list.append(class_idx)
                    valid_count += 1

        logger.info(f"  [{class_name}] -> {valid_count} vectores extraídos.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    logger.info(f"Dataset consolidado en RAM: {X.shape[0]} muestras de {X.shape[1]} dimensiones.")
    
    logger.info("Entrenando Pipeline: StandardScaler -> CalibratedClassifierCV(LinearSVC)...")
    
    base_svc = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42)
    calibrated_svc = CalibratedClassifierCV(base_svc, method='sigmoid', cv=5)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', calibrated_svc)
    ])
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pipeline.fit(X, y)
        
    # Calcular exactitud rápida sobre el mismo set para confirmar convergencia
    preds = pipeline.predict(X)
    acc = np.mean(preds == y)
    logger.info(f"Entrenamiento completado. Accuracy de entrenamiento: {acc*100:.2f}%")

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_pickle({
        "model": pipeline,
        "class_names": class_names,
    }, OUTPUT_MODEL_PATH)
    logger.info(f"[OK] Modelo definitivo Re-ID guardado en {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()