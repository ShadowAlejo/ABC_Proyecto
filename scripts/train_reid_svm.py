"""Entrena el SVM Re-ID (LBP-U por bloques) a partir de dataset/raw_images/<sujeto>/*.jpg."""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_extraction.body.torso_roi_isolator import isolate_torso_roi
from feature_extraction.body.stable_zone_masker import apply_stable_zone_mask
from feature_extraction.body.spatial_grid_histogram import extract_spatial_grid_lbp
from retraining.class_balancer import compute_class_weights
from retraining.cross_validation_runner import CrossValidationRunner
from retraining.model_promotion_gatekeeper import ModelPromotionGatekeeper
from sklearn.svm import SVC
from utils.file_io_helpers import list_files, save_pickle
from utils.logger import get_logger

logger = get_logger("train_reid_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_reid/svm_reid_model.pkl")


def build_dataset():
    class_names = sorted([d.name for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise RuntimeError(f"No se encontraron carpetas de sujetos en {RAW_IMAGES_DIR}")

    X, y = [], []
    for class_idx, class_name in enumerate(class_names):
        image_paths = list_files(RAW_IMAGES_DIR / class_name, (".jpg", ".jpeg", ".png"))
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                torso_roi = isolate_torso_roi(img)  # redimensiona a 64x128 [REQ-RID-02]
                gray_roi = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
                _, weight_map = apply_stable_zone_mask(torso_roi)
                lbp_vector = extract_spatial_grid_lbp(gray_roi, weight_map=weight_map)  # [REQ-RID-04/05]
            except ValueError as e:
                logger.warning(f"Omitiendo {img_path.name}: {e}")
                continue

            X.append(lbp_vector)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), class_names


def main():
    X, y, class_names = build_dataset()
    logger.info(f"Dataset Re-ID construido: {X.shape[0]} muestras, {len(class_names)} clases.")

    class_weights = compute_class_weights(y)
    model = SVC(kernel="linear", C=1.0, class_weight=class_weights, probability=False)

    cv_runner = CrossValidationRunner(method="kfold", n_splits=5)
    cv_result = cv_runner.evaluate(model, X, y)
    logger.info(f"Resultado CV Re-ID: accuracy={cv_result.mean_accuracy:.4f}, f1_macro={cv_result.mean_f1_macro:.4f}")

    gatekeeper = ModelPromotionGatekeeper()
    decision = gatekeeper.evaluate_promotion(cv_result, None)
    logger.info(f"Decisión de promoción Re-ID: {decision.promote} — {decision.reason}")

    if not decision.promote:
        logger.error("El modelo Re-ID NO cumple el umbral mínimo de calidad. No se guardará.")
        return

    model.fit(X, y)
    save_pickle({"model": model, "class_names": class_names}, OUTPUT_MODEL_PATH)
    logger.info(f"Modelo SVM Re-ID guardado en: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()