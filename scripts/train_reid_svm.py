"""Entrena el SVM Re-ID (LBP-U multi-escala) a partir de dataset/raw_images/<sujeto>/*.jpg.

Mejoras aplicadas:
  - Extracción multi-variante fotométrica (White-Patch + CLAHE en espacio LAB).
  - Augmentación sintética para Re-ID SIN rotaciones para preservar bandas anatómicas e histogramas LBP.
  - Validación Cruzada Fold-Aware (Zero Data Leakage): evaluación 100% pura sobre imágenes reales.
  - Búsqueda de hiperparámetro C con 5-fold CV y promoción con ModelPromotionGatekeeper.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_extraction.body.torso_roi_isolator import isolate_torso_roi
from feature_extraction.body.stable_zone_masker import apply_stable_zone_mask
from feature_extraction.body.spatial_grid_histogram import extract_spatial_grid_lbp
from retraining.class_balancer import compute_class_weights
from retraining.data_augmentation_engine import DataAugmentationEngine
from retraining.cross_validation_runner import CrossValidationResult
from retraining.model_promotion_gatekeeper import ModelPromotionGatekeeper
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from utils.file_io_helpers import list_files, save_pickle
from utils.logger import get_logger

logger = get_logger("train_reid_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_reid/svm_reid_model.pkl")
CACHE_PATH = Path("dataset/cache/reid_features.npz")
MIN_SAMPLES_FOR_AUGMENTATION = 20


def _extract_reid_vector(img: np.ndarray) -> np.ndarray | None:
    """Aísla torso con enhancement, genera máscara sigmoidea y extrae vector LBP multi-escala."""
    try:
        torso_roi = isolate_torso_roi(img, enhance=True)
        gray_roi = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
        _, weight_map = apply_stable_zone_mask(torso_roi)
        return extract_spatial_grid_lbp(gray_roi, weight_map=weight_map)
    except Exception as e:
        logger.debug(f"Error extrayendo vector Re-ID: {e}")
        return None


def extract_reid_worker(task_info: tuple[int, Path, str, int]) -> tuple[int, int, str, np.ndarray | None, list[np.ndarray]]:
    """Trabajador individual: extrae vector real + 2 variantes fotométricas."""
    img_global_id, img_path, class_name, class_idx = task_info
    img = cv2.imread(str(img_path))
    if img is None:
        return img_global_id, class_idx, str(img_path), None, []

    vec_real = _extract_reid_vector(img)
    if vec_real is None:
        return img_global_id, class_idx, str(img_path), None, []

    augmenter = DataAugmentationEngine()
    aug_vectors = []
    photo_samples = augmenter.generate_reid_photometric_samples(img, n_samples=15)
    for p_img in photo_samples:
        p_vec = _extract_reid_vector(p_img)
        if p_vec is not None:
            aug_vectors.append(p_vec)

    return img_global_id, class_idx, str(img_path), vec_real, aug_vectors


def build_dataset(augmenter: DataAugmentationEngine):
    if CACHE_PATH.exists():
        logger.info(f"Cargando características Re-ID desde caché: {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        real_X = data['real_X']
        real_y = data['real_y']
        real_img_ids = data['real_img_ids']
        aug_dict = data['aug_dict'].item()
        img_paths_dict = data['img_paths_dict'].item()
        class_names = data['class_names'].tolist()
        return real_X, real_y, real_img_ids, aug_dict, img_paths_dict, class_names

    class_names_all = sorted([d.name for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    if len(class_names_all) == 0:
        raise RuntimeError(f"No se encontraron carpetas de sujetos en {RAW_IMAGES_DIR}")

    logger.info(f"Clases Re-ID detectadas ({len(class_names_all)}): {class_names_all}")

    # 1. Fase Map: Descubrimiento de archivos con balance de hasta 450 fotos por clase
    tasks = []
    global_id = 0
    for class_idx, class_name in enumerate(class_names_all):
        image_paths = list_files(RAW_IMAGES_DIR / class_name, (".jpg", ".jpeg", ".png"))
        image_paths.sort()
        # Cap a 450 imágenes para balance óptimo
        selected_paths = image_paths[:450]
        for img_path in selected_paths:
            tasks.append((global_id, img_path, class_name, class_idx))
            global_id += 1

    logger.info(f"Fase Map completada: {len(tasks)} imágenes seleccionadas para procesar.")

    # 2. Fase Distribución y Ejecución Paralela
    num_workers = max(1, min(8, os.cpu_count() - 1))
    logger.info(f"Lanzando {num_workers} procesos trabajadores para extracción Re-ID...")
    
    extracted_features = []
    batch_size = 200
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(extract_reid_worker, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            img_id, c_idx, path_str, vec_real, aug_vecs = res
            if vec_real is not None:
                extracted_features.append(res)
            completed += 1
            if completed % batch_size == 0 or completed == len(tasks):
                logger.info(f"Progreso extracción Re-ID: {completed}/{len(tasks)} procesadas.")

    # 3. Fase Reduce
    class_results = defaultdict(list)
    for img_global_id, class_idx, img_path, vec_real, aug_vectors in extracted_features:
        class_results[class_idx].append((img_global_id, img_path, vec_real, aug_vectors))

    real_X_list, real_y_list, real_img_ids_list = [], [], []
    aug_dict = {}
    img_paths_dict = {}
    valid_class_names = []

    for old_class_idx, class_name in enumerate(class_names_all):
        results = class_results.get(old_class_idx, [])
        if len(results) < 10:
            logger.warning(f"Sin suficientes vectores válidos para la clase Re-ID '{class_name}', se omite.")
            continue

        valid_class_names.append(class_name)
        new_class_idx = len(valid_class_names) - 1

        for img_global_id, img_path, vec_real, aug_vectors in results:
            real_X_list.append(vec_real)
            real_y_list.append(new_class_idx)
            real_img_ids_list.append(img_global_id)
            aug_dict[img_global_id] = aug_vectors
            img_paths_dict[img_global_id] = img_path

        logger.info(f"[{class_name}] -> {len(results)} imágenes reales (+{len(results)*2} variantes fotométricas).")

    real_X = np.array(real_X_list, dtype=np.float32)
    real_y = np.array(real_y_list, dtype=np.int64)
    real_img_ids = np.array(real_img_ids_list, dtype=np.int64)

    # Caching
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        real_X=real_X,
        real_y=real_y,
        real_img_ids=real_img_ids,
        aug_dict=np.array(aug_dict, dtype=object),
        img_paths_dict=np.array(img_paths_dict, dtype=object),
        class_names=valid_class_names
    )
    logger.info(f"Características Re-ID guardadas en caché: {CACHE_PATH}")

    return real_X, real_y, real_img_ids, aug_dict, img_paths_dict, valid_class_names


def main():
    from sklearn.svm import LinearSVC
    augmenter = DataAugmentationEngine()
    real_X, real_y, real_img_ids, aug_dict, img_paths_dict, class_names = build_dataset(augmenter)
    logger.info(f"Dataset Re-ID construido: {real_X.shape[0]} imágenes reales x {real_X.shape[1]} dimensiones, {len(class_names)} clases.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_c = 1.0
    best_result = None

    # Búsqueda de hiperparámetro C mediante 5-Fold CV Fold-Aware (Zero Data Leakage)
    for c_val in [0.1, 1.0, 10.0]:
        fold_accuracies = []
        fold_f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(real_X, real_y)):
            # 1. Construir conjunto de entrenamiento del Fold con Augmentation fotométrica
            X_train_fold, y_train_fold = [], []

            for idx in train_idx:
                X_train_fold.append(real_X[idx])
                cls = real_y[idx]
                y_train_fold.append(cls)
                img_id = int(real_img_ids[idx])
                for aug_vec in aug_dict.get(img_id, []):
                    X_train_fold.append(aug_vec)
                    y_train_fold.append(cls)

            # 2. Conjunto de validación del Fold: 100% PURO sobre imágenes reales
            X_val_fold = real_X[val_idx]
            y_val_fold = real_y[val_idx]

            X_train_arr = np.array(X_train_fold, dtype=np.float32)
            y_train_arr = np.array(y_train_fold, dtype=np.int64)

            fold_model = LinearSVC(C=c_val, class_weight="balanced", dual=False, max_iter=2000, random_state=42)
            fold_model.fit(X_train_arr, y_train_arr)

            preds = fold_model.predict(X_val_fold)
            fold_accuracies.append(accuracy_score(y_val_fold, preds))
            fold_f1_scores.append(f1_score(y_val_fold, preds, average="macro", zero_division=0))

        cv_res = CrossValidationResult(
            mean_accuracy=float(np.mean(fold_accuracies)),
            mean_f1_macro=float(np.mean(fold_f1_scores)),
            fold_scores=fold_accuracies
        )
        logger.info(f"Evaluación Fold-Aware C={c_val}: accuracy={cv_res.mean_accuracy:.4f}, f1_macro={cv_res.mean_f1_macro:.4f}")

        if best_result is None or cv_res.mean_f1_macro > best_result.mean_f1_macro:
            best_result = cv_res
            best_c = c_val

    logger.info(f"Parámetro C óptimo seleccionado: {best_c} (f1_macro={best_result.mean_f1_macro:.4f})")

    gatekeeper = ModelPromotionGatekeeper()
    decision = gatekeeper.evaluate_promotion(best_result, None)
    logger.info(f"Decisión de promoción Re-ID: {decision.promote} — {decision.reason}")

    if not decision.promote:
        logger.error("El modelo Re-ID NO cumple el umbral mínimo de calidad. No se guardará.")
        return

    # Entrenamiento del modelo final con todos los datos y augmentación completa
    X_full, y_full = [], []
    for idx in range(len(real_X)):
        X_full.append(real_X[idx])
        cls = real_y[idx]
        y_full.append(cls)
        img_id = int(real_img_ids[idx])
        for aug_vec in aug_dict.get(img_id, []):
            X_full.append(aug_vec)
            y_full.append(cls)

    X_full = np.array(X_full, dtype=np.float32)
    y_full = np.array(y_full, dtype=np.int64)

    final_model = LinearSVC(C=best_c, class_weight="balanced", dual=False, max_iter=2000, random_state=42)
    final_model.fit(X_full, y_full)

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_pickle({"model": final_model, "class_names": class_names}, OUTPUT_MODEL_PATH)
    logger.info(f"Modelo SVM Re-ID optimizado guardado en: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()