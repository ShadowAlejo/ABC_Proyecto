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
    group_list = []
    is_real_list = []
    
    global_group_idx = 0

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
                group_list.append(global_group_idx)
                is_real_list.append(True)
                valid_count += 1
                
            # 20 Variantes geométricas
            variants = augmenter.generate_reid_geometric_samples(img, n_samples=20)
            for v_img in variants:
                v_vec = _extract_lbp(v_img)
                if v_vec is not None:
                    X_list.append(v_vec)
                    y_list.append(class_idx)
                    group_list.append(global_group_idx)
                    is_real_list.append(False)
                    valid_count += 1
                    
            global_group_idx += 1

        logger.info(f"  [{class_name}] -> {valid_count} vectores extraídos.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(group_list, dtype=np.int64)
    is_real = np.array(is_real_list, dtype=bool)

    logger.info(f"Dataset consolidado en RAM: {X.shape[0]} muestras de {X.shape[1]} dimensiones.")
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

    logger.info("Iniciando validación cruzada (Zero Data Leakage)...")
    valid_groups = np.unique(groups)
    group_y = [y[np.where(groups == g)[0][0]] for g in valid_groups]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    custom_cv = []
    for train_g_idx, val_g_idx in skf.split(valid_groups, group_y):
        train_groups = valid_groups[train_g_idx]
        val_groups = valid_groups[val_g_idx]
        train_idx = np.where(np.isin(groups, train_groups))[0]
        val_idx = np.where(np.isin(groups, val_groups) & is_real)[0]
        custom_cv.append((train_idx, val_idx))

    cv_preds = np.zeros(len(X), dtype=np.int64) - 1
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for train_idx, val_idx in custom_cv:
            fold_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42))
            ])
            fold_pipe.fit(X[train_idx], y[train_idx])
            cv_preds[val_idx] = fold_pipe.predict(X[val_idx])
            
    val_mask = cv_preds != -1
    y_val_real = y[val_mask]
    preds_val_real = cv_preds[val_mask]

    logger.info("=== Métricas Finales (Cross-Validation sobre imágenes reales) ===")
    logger.info(f"Accuracy: {accuracy_score(y_val_real, preds_val_real):.4f}")
    logger.info(f"Precision (macro): {precision_score(y_val_real, preds_val_real, average='macro', zero_division=0):.4f}")
    logger.info(f"Recall (macro): {recall_score(y_val_real, preds_val_real, average='macro', zero_division=0):.4f}")
    logger.info(f"F1-Score (macro): {f1_score(y_val_real, preds_val_real, average='macro', zero_division=0):.4f}")
    logger.info(f"Balanced Accuracy: {balanced_accuracy_score(y_val_real, preds_val_real):.4f}")

    logger.info("Entrenando modelo final 100% de datos...")
    
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42))
    ])
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        final_pipeline.fit(X, y)
        calibrator = CalibratedClassifierCV(final_pipeline.named_steps['classifier'], cv='prefit', method='sigmoid')
        calibrator.fit(final_pipeline.named_steps['scaler'].transform(X[is_real]), y[is_real])
        final_pipeline.steps[-1] = ('classifier', calibrator)

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_pickle({
        "model": final_pipeline,
        "class_names": class_names,
    }, OUTPUT_MODEL_PATH)
    logger.info(f"[OK] Modelo definitivo Re-ID guardado en {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()