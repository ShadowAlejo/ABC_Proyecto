"""Entrena el SVM Facial (16 clases) a partir de dataset/raw_images/<sujeto>/*.jpg.

Estrategia de extracción de muestras (máximo recall):
  Capa 1 — detect_training() + validate_face_quality(training_mode=True):
           imagen limpia que pasa los filtros relajados → ACEPTADA.
  Capa 2 — detect_training() recovered + validate_face_quality(training_mode=True):
           rostro encontrado solo con preprocesamiento adicional, pasa filtro relajado → RECUPERADA.
  Capa 3 — sin detección válida → DESCARTADA (irrecuperable).

En todos los casos aceptados, normalize_face(enhance_for_training=True) aplica
CLAHE + unsharp masking antes de calcular HOG para mayor robustez del descriptor.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))  # permite importar módulos del proyecto

from feature_extraction.face.yunet_face_detector import YuNetFaceDetector
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from feature_extraction.face.face_quality_validator import validate_face_quality
from retraining.class_balancer import compute_class_weights
from retraining.data_augmentation_engine import DataAugmentationEngine
from retraining.cross_validation_runner import CrossValidationRunner
from retraining.model_promotion_gatekeeper import ModelPromotionGatekeeper
from sklearn.svm import SVC
from utils.file_io_helpers import list_files, save_pickle, load_pickle
from utils.logger import get_logger

logger = get_logger("train_facial_svm")

RAW_IMAGES_DIR = Path("dataset/raw_images")
OUTPUT_MODEL_PATH = Path("dataset/models/svm_facial/svm_facial_model.pkl")
MIN_SAMPLES_FOR_AUGMENTATION = 20  # clases con menos fotos que esto reciben aumento sintético


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de vector HOG desde una imagen y resultado de detección
# ─────────────────────────────────────────────────────────────────────────────
def _extract_vector(img: np.ndarray, face_result) -> np.ndarray | None:
    """Normaliza el rostro con enhancement y extrae el descriptor HOG.
    Devuelve None si el crop resulta inválido."""
    face_gray = normalize_face(img, face_result, enhance_for_training=True)
    if face_gray is None:
        return None
    return extract_hog_features(face_gray)


# ─────────────────────────────────────────────────────────────────────────────
# Intento de extracción para imágenes sintéticas de augmentación
# ─────────────────────────────────────────────────────────────────────────────
def _extract_from_synthetic(img: np.ndarray, detector: YuNetFaceDetector) -> np.ndarray | None:
    """Usa detect_training() para extraer HOG de imágenes sintéticas augmentadas."""
    face_result = detector.detect_training(img)
    if not face_result.detected:
        return None
    return _extract_vector(img, face_result)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del dataset con 3 capas de recovery
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(face_detector: YuNetFaceDetector, augmenter: DataAugmentationEngine):
    class_names = sorted([d.name for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise RuntimeError(f"No se encontraron carpetas de sujetos en {RAW_IMAGES_DIR}")

    logger.info(f"Clases detectadas ({len(class_names)}): {class_names}")

    X, y = [], []
    total_accepted = total_recovered = total_discarded = 0

    for class_idx, class_name in enumerate(class_names):
        image_paths = list_files(RAW_IMAGES_DIR / class_name, (".jpg", ".jpeg", ".png"))
        if not image_paths:
            logger.warning(f"Sin imágenes para la clase '{class_name}', se omite.")
            continue

        class_vectors = []
        accepted = recovered = discarded = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                discarded += 1
                continue

            # ── Capa 1 & 2: detect_training() con cascada de 8 preprocesados ──
            face_result = face_detector.detect_training(img)

            if not face_result.detected:
                # Capa 3: irrecuperable
                discarded += 1
                logger.debug(
                    f"[DESCARTADA] {class_name}/{img_path.name} — "
                    "sin detección válida tras todas las variantes."
                )
                continue

            # Validación de calidad en modo entrenamiento (umbrales relajados)
            quality = validate_face_quality(img, face_result, training_mode=True)
            if not quality.is_valid:
                discarded += 1
                logger.debug(
                    f"[DESCARTADA] {class_name}/{img_path.name} — "
                    f"calidad insuficiente: {quality.reasons}"
                )
                continue

            vector = _extract_vector(img, face_result)
            if vector is None:
                discarded += 1
                continue

            class_vectors.append((vector, img))

            if face_result.was_recovered:
                recovered += 1
                logger.debug(
                    f"[RECUPERADA] {class_name}/{img_path.name} — "
                    f"conf={face_result.confidence:.3f}, sharpness={quality.sharpness:.1f}"
                )
            else:
                accepted += 1

        # ── Aumento sintético si la clase tiene pocas muestras [REQ-ENT-02] ──
        total_real = len(class_vectors)
        if 0 < total_real < MIN_SAMPLES_FOR_AUGMENTATION:
            needed = MIN_SAMPLES_FOR_AUGMENTATION - total_real
            logger.info(
                f"Clase '{class_name}': {total_real} muestras reales → "
                f"generando {needed} sintéticas."
            )
            base_img = class_vectors[0][1]
            synth_ok = 0
            for synthetic_img in augmenter.generate_synthetic_samples(base_img, n_samples=needed):
                vec = _extract_from_synthetic(synthetic_img, face_detector)
                if vec is not None:
                    class_vectors.append((vec, synthetic_img))
                    synth_ok += 1
            logger.info(f"  → {synth_ok}/{needed} sintéticas incorporadas.")

        for vector, _ in class_vectors:
            X.append(vector)
            y.append(class_idx)

        total_accepted += accepted
        total_recovered += recovered
        total_discarded += discarded

        logger.info(
            f"[{class_name}] "
            f"✅ aceptadas={accepted}  "
            f"🔄 recuperadas={recovered}  "
            f"❌ descartadas={discarded}  "
            f"→ {len(class_vectors)} vectores"
        )

    # Resumen global
    logger.info("─" * 60)
    logger.info(f"RESUMEN GLOBAL DEL DATASET:")
    logger.info(f"  ✅ Aceptadas (directas)  : {total_accepted}")
    logger.info(f"  🔄 Recuperadas (preproc.) : {total_recovered}")
    logger.info(f"  ❌ Descartadas            : {total_discarded}")
    logger.info(
        f"  Tasa de cobertura: "
        f"{(total_accepted + total_recovered) / max(1, total_accepted + total_recovered + total_discarded) * 100:.1f}%"
    )
    logger.info("─" * 60)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), class_names


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento principal
# ─────────────────────────────────────────────────────────────────────────────
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