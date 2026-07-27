"""Entrena el SVM Facial (16 clases) a partir de dataset/raw_images/<sujeto>/*.jpg."""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))  # permite importar módulos del proyecto

from feature_extraction.face.yunet_face_detector import YuNetFaceDetector
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from retraining.class_balancer import compute_class_weights
from retraining.data_augmentation_engine import DataAugmentationEngine
from retraining.cross_validation_runner import CrossValidationRunner
from retraining.model_promotion_gatekeeper import ModelPromotionGatekeeper
from sklearn.svm import SVC
from utils.file_io_helpers import list_files, save_pickle, load_pickle
from utils.logger import get_logger
from feature_extraction.face.face_quality_validator import validate_face_quality

logger = get_logger("train_facial_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_facial/svm_facial_model.pkl")
MIN_SAMPLES_FOR_AUGMENTATION = 20  # clases con menos fotos que esto reciben aumento sintético


def build_dataset(face_detector: YuNetFaceDetector, augmenter: DataAugmentationEngine):
    class_names = sorted([d.name for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise RuntimeError(f"No se encontraron carpetas de sujetos en {RAW_IMAGES_DIR}")

    logger.info(f"Clases detectadas ({len(class_names)}): {class_names}")

    X, y = [], []
    for class_idx, class_name in enumerate(class_names):
        image_paths = list_files(RAW_IMAGES_DIR / class_name, (".jpg", ".jpeg", ".png"))
        if not image_paths:
            logger.warning(f"Sin imágenes para la clase '{class_name}', se omite.")
            continue

        class_vectors = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            face_result = face_detector.detect(img)
            quality = validate_face_quality(img, face_result)

            if not quality.is_valid:
                logger.warning(f"Omitiendo {img_path.name} ({class_name}): {quality.reasons}")
                continue

            face_gray = normalize_face(img, face_result)
            if face_gray is None:
                continue
            class_vectors.append((extract_hog_features(face_gray), img))

        # Aumento sintético si la clase tiene pocas muestras [REQ-ENT-02]
        if 0 < len(class_vectors) < MIN_SAMPLES_FOR_AUGMENTATION:
            needed = MIN_SAMPLES_FOR_AUGMENTATION - len(class_vectors)
            logger.info(f"Clase '{class_name}' con {len(class_vectors)} muestras; generando {needed} sintéticas.")
            base_img = class_vectors[0][1]
            for synthetic_img in augmenter.generate_synthetic_samples(base_img, n_samples=needed):
                face_result = face_detector.detect(synthetic_img)
                face_gray = normalize_face(synthetic_img, face_result)
                if face_gray is not None:
                    class_vectors.append((extract_hog_features(face_gray), synthetic_img))

        for vector, _ in class_vectors:
            X.append(vector)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), class_names


def main():
    face_detector = YuNetFaceDetector()
    augmenter = DataAugmentationEngine()

    X, y, class_names = build_dataset(face_detector, augmenter)
    logger.info(f"Dataset construido: {X.shape[0]} muestras, {len(class_names)} clases.")

    class_weights = compute_class_weights(y)  # [REQ-ENT-01]
    logger.info(f"Pesos de clase (class_weight='balanced'): {class_weights}")

    model = SVC(kernel="linear", C=1.0, class_weight=class_weights, probability=False)

    cv_runner = CrossValidationRunner(method="kfold", n_splits=5)  # [REQ-ENT-05]
    cv_result = cv_runner.evaluate(model, X, y)
    logger.info(f"Resultado CV: accuracy={cv_result.mean_accuracy:.4f}, f1_macro={cv_result.mean_f1_macro:.4f}")

    gatekeeper = ModelPromotionGatekeeper()
    production_result = None
    if OUTPUT_MODEL_PATH.exists():
        # Si ya existe un modelo, podrías re-evaluarlo aquí para comparar (omitido por simplicidad).
        pass

    decision = gatekeeper.evaluate_promotion(cv_result, production_result)
    logger.info(f"Decisión de promoción: {decision.promote} — {decision.reason}")

    if not decision.promote:
        logger.error("El modelo NO cumple el umbral mínimo de calidad. No se guardará.")
        return

    model.fit(X, y)  # entrenamiento final con el 100% de los datos
    save_pickle({"model": model, "class_names": class_names}, OUTPUT_MODEL_PATH)
    logger.info(f"Modelo SVM Facial guardado en: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()