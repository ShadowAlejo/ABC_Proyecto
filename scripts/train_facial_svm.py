"""Entrena el Ensamble SVM Facial (Global, Ojos, Nariz/Boca) con HOG en canal único e iluminación Tan & Triggs.

Optimizaciones aplicadas:
  - Memory-Mapped Out-of-Core Processing para manejar 100K+ vectores sin agotar la RAM.
  - Generación de Aumentación paralela (15 variaciones sintéticas) escritas directamente a disco.
  - Validación Cruzada Fold-Aware estricta (Zero Data Leakage).
  - Entrenamiento por mini-batches con `partial_fit` para estabilizar el Descenso de Gradiente Estocástico.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import warnings
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classification.subspace_facial_ensemble import SubspaceFacialEnsemble
from feature_extraction.face.yoloface_detector import YoloFaceDetector
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from feature_extraction.face.face_quality_validator import validate_face_quality
from retraining.data_augmentation_engine import DataAugmentationEngine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score as sk_f1
from utils.file_io_helpers import list_files, save_pickle, load_image
from utils.logger import get_logger

logger = get_logger("train_facial_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_facial/svm_facial_model.pkl")
MEMMAP_DIR = Path("dataset/cache/memmap")

# Constantes de Extracción
AUG_COUNT = 20
TOTAL_PER_IMAGE = 1 + AUG_COUNT
DIMS_G = 1200
DIMS_U = 1680
DIMS_L = 960

def extract_worker(task_info: tuple[int, Path, int, Path]):
    """Trabajador paralelo: extrae características e inserta en memmap."""
    cv2.setNumThreads(1)
    start_idx, img_path, class_idx, memmap_dir = task_info
    
    # 1. Cargar imagen original
    img_bgr = load_image(img_path)
    if img_bgr is None:
        return start_idx, False

    # Reducir imágenes gigantes
    h, w = img_bgr.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    detector = YoloFaceDetector()
    det_res = detector.detect_training(img_bgr)
    if not det_res.detected or det_res.bbox is None:
        return start_idx, False
    
    q_report = validate_face_quality(img_bgr, det_res, training_mode=True)
    if not q_report.is_valid:
        return start_idx, False

    norm_face, _ = normalize_face(img_bgr, det_res)
    if norm_face is None:
        return start_idx, False
    
    feat_g, feat_u, feat_l = extract_hog_features(norm_face)

    # Abrir memmaps locales
    mm_g = np.memmap(memmap_dir / "X_g.dat", dtype='float32', mode='r+')
    mm_u = np.memmap(memmap_dir / "X_u.dat", dtype='float32', mode='r+')
    mm_l = np.memmap(memmap_dir / "X_l.dat", dtype='float32', mode='r+')
    mm_y = np.memmap(memmap_dir / "y.dat", dtype='int64', mode='r+')
    mm_real = np.memmap(memmap_dir / "is_real.dat", dtype='bool', mode='r+')
    mm_group = np.memmap(memmap_dir / "group.dat", dtype='int64', mode='r+')

    # Determinar shape
    n_rows = mm_g.shape[0] // DIMS_G
    mm_g = mm_g.reshape((n_rows, DIMS_G))
    mm_u = mm_u.reshape((n_rows, DIMS_U))
    mm_l = mm_l.reshape((n_rows, DIMS_L))

    # Escribir el Real (índice start_idx)
    mm_g[start_idx] = feat_g
    mm_u[start_idx] = feat_u
    mm_l[start_idx] = feat_l
    mm_y[start_idx] = class_idx
    mm_real[start_idx] = True
    mm_group[start_idx] = start_idx // TOTAL_PER_IMAGE

    # Escribir Aumentaciones
    augmenter = DataAugmentationEngine()
    for i in range(AUG_COUNT):
        curr_idx = start_idx + 1 + i
        jittered_landmarks = augmenter.jitter_landmarks(det_res.landmarks, scale=0.02) if det_res.landmarks is not None else None
        aug_face, _ = normalize_face(img_bgr, det_res, custom_landmarks=jittered_landmarks)
        if aug_face is None:
            continue
        aug_face = augmenter.apply_random_erasing(aug_face, sl=0.02, sh=0.15, r1=0.3, r2=3.3)
        af_g, af_u, af_l = extract_hog_features(aug_face)
        
        mm_g[curr_idx] = af_g
        mm_u[curr_idx] = af_u
        mm_l[curr_idx] = af_l
        mm_y[curr_idx] = class_idx
        mm_real[curr_idx] = False
        mm_group[curr_idx] = start_idx // TOTAL_PER_IMAGE

    return start_idx, True


def main():
    logger.info("Iniciando preparación Out-of-Core de dataset...")
    MEMMAP_DIR.mkdir(parents=True, exist_ok=True)
    
    person_dirs = [d for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in person_dirs])
    
    if len(class_names) < 2:
        logger.error("Se necesitan al menos 2 clases para entrenar.")
        return

    # Fase Map: Seleccionar archivos
    tasks = []
    global_idx = 0
    for class_idx, class_name in enumerate(class_names):
        p_dir = RAW_IMAGES_DIR / class_name
        files = list_files(p_dir, extensions=[".jpg", ".jpeg", ".png"])
        files.sort()
        selected_files = files[:450]
        for f in selected_files:
            tasks.append((global_idx, f, class_idx, MEMMAP_DIR))
            global_idx += TOTAL_PER_IMAGE

    n_total = global_idx
    logger.info(f"Asignando archivos binarios para {n_total} vectores ({n_total//TOTAL_PER_IMAGE} reales).")

    # Pre-allocating memmaps (w+)
    mm_g = np.memmap(MEMMAP_DIR / "X_g.dat", dtype='float32', mode='w+', shape=(n_total, DIMS_G))
    mm_u = np.memmap(MEMMAP_DIR / "X_u.dat", dtype='float32', mode='w+', shape=(n_total, DIMS_U))
    mm_l = np.memmap(MEMMAP_DIR / "X_l.dat", dtype='float32', mode='w+', shape=(n_total, DIMS_L))
    mm_y = np.memmap(MEMMAP_DIR / "y.dat", dtype='int64', mode='w+', shape=(n_total,))
    mm_real = np.memmap(MEMMAP_DIR / "is_real.dat", dtype='bool', mode='w+', shape=(n_total,))
    mm_group = np.memmap(MEMMAP_DIR / "group.dat", dtype='int64', mode='w+', shape=(n_total,))
    
    # Initialize with -1 to detect unwritten rows
    mm_y[:] = -1
    
    # Flush (ensure creation)
    del mm_g, mm_u, mm_l, mm_y, mm_real, mm_group

    num_workers = max(1, min(8, os.cpu_count() - 1))
    logger.info(f"Extrayendo y mapeando características en disco (procesos: {num_workers})...")
    
    valid_groups = []
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(extract_worker, t): t for t in tasks}
        for future in as_completed(futures):
            start_idx, success = future.result()
            if success:
                valid_groups.append(start_idx // TOTAL_PER_IMAGE)
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                logger.info(f"Progreso extracción: {completed}/{len(tasks)} imágenes procesadas.")

    valid_groups = np.array(valid_groups, dtype=np.int64)
    logger.info(f"Extracción completada. Imágenes reales válidas: {len(valid_groups)}")

    # Re-abrir para lectura (r)
    mm_g = np.memmap(MEMMAP_DIR / "X_g.dat", dtype='float32', mode='r', shape=(n_total, DIMS_G))
    mm_u = np.memmap(MEMMAP_DIR / "X_u.dat", dtype='float32', mode='r', shape=(n_total, DIMS_U))
    mm_l = np.memmap(MEMMAP_DIR / "X_l.dat", dtype='float32', mode='r', shape=(n_total, DIMS_L))
    mm_y = np.memmap(MEMMAP_DIR / "y.dat", dtype='int64', mode='r', shape=(n_total,))
    mm_real = np.memmap(MEMMAP_DIR / "is_real.dat", dtype='bool', mode='r', shape=(n_total,))
    mm_group = np.memmap(MEMMAP_DIR / "group.dat", dtype='int64', mode='r', shape=(n_total,))

    # Obtener etiquetas de cada grupo
    group_y = np.array([mm_y[g * TOTAL_PER_IMAGE] for g in valid_groups])

    logger.info("Iniciando Validación Cruzada Zero Data Leakage...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold_idx, (train_group_idx, val_group_idx) in enumerate(skf.split(valid_groups, group_y)):
        train_groups = valid_groups[train_group_idx]
        val_groups = valid_groups[val_group_idx]
        
        # Filtros de índices en memmap
        train_mask = np.isin(mm_group, train_groups) & (mm_y != -1)
        # Validación SOLO reales
        val_mask = np.isin(mm_group, val_groups) & mm_real & (mm_y != -1)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        # Mezclar entrenamiento para SGD
        np.random.shuffle(train_indices)
        
        fold_model = SubspaceFacialEnsemble()
        
        # Batch fitting
        batch_size = 32000
        n_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size != 0 else 0)
        
        # Scikit-learn classes
        unique_classes = np.unique(group_y)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for b in range(n_batches):
                b_idx = train_indices[b*batch_size : (b+1)*batch_size]
                X_batch_tuple = (mm_g[b_idx], mm_u[b_idx], mm_l[b_idx])
                y_batch = mm_y[b_idx]
                fold_model.partial_fit(X_batch_tuple, y_batch, classes=unique_classes)
            
            fold_model.calibrate((mm_g[val_indices], mm_u[val_indices], mm_l[val_indices]), mm_y[val_indices])
            
            # Evaluación
            X_val_tuple = (mm_g[val_indices], mm_u[val_indices], mm_l[val_indices])
            y_val = mm_y[val_indices]
            preds = fold_model.predict(X_val_tuple)

        f1 = sk_f1(y_val, preds, average="macro", zero_division=0)
        acc = float(np.mean(preds == y_val))
        fold_scores.append(f1)
        logger.info(f"  Fold {fold_idx + 1}/5 -> Acc: {acc*100:.2f}%, F1: {f1:.4f} (Train: {len(train_indices)} aug, Val: {len(val_indices)} reales)")

    mean_f1 = float(np.mean(fold_scores))
    logger.info(f"F1-Macro CV Global (Zero Data Leakage): {mean_f1:.4f}")

    if mean_f1 >= 0.80:
        logger.info("Entrenando modelo final con 100% de datos en modo Mini-Batch...")
        all_train_mask = (mm_y != -1)
        all_train_indices = np.where(all_train_mask)[0]
        np.random.shuffle(all_train_indices)
        
        final_model = SubspaceFacialEnsemble()
        batch_size = 32000
        n_batches = len(all_train_indices) // batch_size + (1 if len(all_train_indices) % batch_size != 0 else 0)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for b in range(n_batches):
                b_idx = all_train_indices[b*batch_size : (b+1)*batch_size]
                X_batch_tuple = (mm_g[b_idx], mm_u[b_idx], mm_l[b_idx])
                y_batch = mm_y[b_idx]
                final_model.partial_fit(X_batch_tuple, y_batch, classes=np.unique(group_y))

        real_mask = mm_real & (mm_y != -1)
        real_indices = np.where(real_mask)[0]
        X_calib = (mm_g[real_indices], mm_u[real_indices], mm_l[real_indices])
        y_calib = mm_y[real_indices]
        
        logger.info("Ajustando calibrador de probabilidades (Platt Scaling)...")
        final_model.calibrate(X_calib, y_calib)

        preds_final = final_model.predict(X_calib)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
        logger.info("=== Métricas Finales (Sobre datos Reales de Calibración) ===")
        logger.info(f"Accuracy: {accuracy_score(y_calib, preds_final):.4f}")
        logger.info(f"Precision (macro): {precision_score(y_calib, preds_final, average='macro', zero_division=0):.4f}")
        logger.info(f"Recall (macro): {recall_score(y_calib, preds_final, average='macro', zero_division=0):.4f}")
        logger.info(f"F1-Score (macro): {f1_score(y_calib, preds_final, average='macro', zero_division=0):.4f}")
        logger.info(f"Balanced Accuracy: {balanced_accuracy_score(y_calib, preds_final):.4f}")

        OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_pickle({
            "model": final_model,
            "class_names": class_names,
            "cv_f1_macro": mean_f1
        }, OUTPUT_MODEL_PATH)
        logger.info(f"[OK] Modelo SVM Facial guardado en {OUTPUT_MODEL_PATH}")
    else:
        logger.warning(f"[ALERTA] El F1-Macro CV ({mean_f1:.4f}) es menor al umbral (0.80). NO se guardará.")

if __name__ == "__main__":
    main()