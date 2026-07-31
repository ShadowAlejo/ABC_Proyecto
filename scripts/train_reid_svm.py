"""Entrena el SVM Re-ID (LBP-U multi-escala) a partir de dataset/raw_images/<sujeto>/*.jpg.

Mejoras aplicadas:
  - Extracción multi-variante (real + variante CLAHE para duplicar representatividad).
  - Normalización de iluminación (White-Patch + CLAHE en torso).
  - Augmentación sintética mediante DataAugmentationEngine para clases minoritarias.
  - Búsqueda de parámetro de regularización C ∈ [0.1, 1.0, 10.0] con 5-fold CV.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_extraction.body.torso_roi_isolator import isolate_torso_roi
from feature_extraction.body.stable_zone_masker import apply_stable_zone_mask
from feature_extraction.body.spatial_grid_histogram import extract_spatial_grid_lbp
from retraining.class_balancer import compute_class_weights
from retraining.data_augmentation_engine import DataAugmentationEngine
from retraining.cross_validation_runner import CrossValidationRunner
from retraining.model_promotion_gatekeeper import ModelPromotionGatekeeper
from sklearn.svm import SVC
from utils.file_io_helpers import list_files, save_pickle
from utils.logger import get_logger

logger = get_logger("train_reid_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_reid/svm_reid_model.pkl")
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


CACHE_PATH = Path("dataset/cache/reid_features.npz")

def _worker_process_reid_chunk(chunk):
    import cv2
    cv2.setNumThreads(0)
    results = []
    for img_path, class_idx in chunk:
        img = cv2.imread(str(img_path))
        if img is None: continue

        # 1. Extracción de imagen real
        vec = _extract_reid_vector(img)
        if vec is None: continue

        # 2. Variante CLAHE directa
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        clahe_img = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

        vec_clahe = _extract_reid_vector(clahe_img)
        results.append((class_idx, img_path, vec, vec_clahe))
    return results


def build_dataset(augmenter: DataAugmentationEngine):
    if CACHE_PATH.exists():
        logger.info(f"Cargando características Re-ID desde caché: {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        return data['X'], data['y'], data['class_names'].tolist()

    class_names = sorted([d.name for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise RuntimeError(f"No se encontraron carpetas de sujetos en {RAW_IMAGES_DIR}")

    logger.info(f"Clases Re-ID detectadas ({len(class_names)}): {class_names}")

    # 1. Fase Map: Lista plana
    flat_tasks = []
    for class_idx, class_name in enumerate(class_names):
        image_paths = list_files(RAW_IMAGES_DIR / class_name, (".jpg", ".jpeg", ".png"))
        for img_path in image_paths:
            flat_tasks.append((img_path, class_idx))

    logger.info(f"Fase Map completada: {len(flat_tasks)} imágenes para procesar.")

    # 2. Fase Distribución y Ejecución (MIMD)
    num_workers = max(1, min(8, os.cpu_count() - 1))
    chunk_size = max(1, len(flat_tasks) // num_workers)
    chunks = [flat_tasks[i:i + chunk_size] for i in range(0, len(flat_tasks), chunk_size)]

    logger.info(f"Lanzando {num_workers} procesos trabajadores para extracción Re-ID...")
    extracted_features = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker_process_reid_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                extracted_features.extend(future.result())
            except Exception as e:
                logger.error(f"Error en trabajador Re-ID: {e}")

    # 3. Fase Reduce y Augmentación Sintética
    from collections import defaultdict
    class_results = defaultdict(list)
    for class_idx, img_path, vec, vec_clahe in extracted_features:
        class_results[class_idx].append((img_path, vec, vec_clahe))

    X, y = [], []
    total_accepted = total_augmented = total_synth = 0

    for class_idx, class_name in enumerate(class_names):
        results = class_results.get(class_idx, [])
        total_real = len(results)
        
        if total_real == 0:
            logger.warning(f"Sin vectores válidos para la clase Re-ID '{class_name}', se omite.")
            continue

        for img_path, vec, vec_clahe in results:
            X.append(vec)
            y.append(class_idx)
            total_accepted += 1
            if vec_clahe is not None:
                X.append(vec_clahe)
                y.append(class_idx)
                total_augmented += 1

        if 0 < total_real < MIN_SAMPLES_FOR_AUGMENTATION:
            needed = MIN_SAMPLES_FOR_AUGMENTATION - total_real
            logger.info(f"Clase Re-ID '{class_name}': {total_real} muestras → generando {needed} sintéticas.")
            base_img = cv2.imread(str(results[0][0])) # Cargar primera imagen para augmentación
            synth_ok = 0
            for synth_img in augmenter.generate_synthetic_samples(base_img, n_samples=needed):
                vec_synth = _extract_reid_vector(synth_img)
                if vec_synth is not None:
                    X.append(vec_synth)
                    y.append(class_idx)
                    synth_ok += 1
            total_synth += synth_ok
            logger.info(f"  → {synth_ok}/{needed} sintéticas incorporadas.")

    logger.info("-" * 60)
    logger.info("RESUMEN GLOBAL DATASET RE-ID:")
    logger.info(f"  [OK] Muestras directas     : {total_accepted}")
    logger.info(f"  [AUG] Variantes de ilum.   : {total_augmented}")
    logger.info(f"  [SYN] Muestras sintéticas  : {total_synth}")
    logger.info("-" * 60)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Caching
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, X=X, y=y, class_names=class_names)
    logger.info(f"Características Re-ID guardadas en caché: {CACHE_PATH}")

    return X, y, class_names


def main():
    augmenter = DataAugmentationEngine()
    X, y, class_names = build_dataset(augmenter)
    logger.info(f"Dataset Re-ID construido: {X.shape[0]} muestras, {len(class_names)} clases. Dimensión vector: {X.shape[1]}")

    class_weights = compute_class_weights(y)
    cv_runner = CrossValidationRunner(method="kfold", n_splits=5)

    best_c = 1.0
    best_result = None

    # Búsqueda rápida de hiperparámetro C
    for c_val in [0.1, 1.0, 10.0]:
        model_test = SVC(kernel="linear", C=c_val, class_weight=class_weights, probability=False)
        cv_res = cv_runner.evaluate(model_test, X, y)
        logger.info(f"Evaluación C={c_val}: accuracy={cv_res.mean_accuracy:.4f}, f1_macro={cv_res.mean_f1_macro:.4f}")
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

    final_model = SVC(kernel="linear", C=best_c, class_weight=class_weights, probability=False)
    final_model.fit(X, y)

    save_pickle({"model": final_model, "class_names": class_names}, OUTPUT_MODEL_PATH)
    logger.info(f"Modelo SVM Re-ID optimizado guardado en: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()