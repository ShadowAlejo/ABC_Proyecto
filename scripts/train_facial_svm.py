"""Entrena el SVM Facial (16 clases) a partir de dataset/raw_images/<sujeto>/*.jpg.

Estrategia de extracción de muestras (máximo recall):
  Capa 1 — detect_training() + validate_face_quality(training_mode=True):
           imagen limpia que pasa los filtros relajados → ACEPTADA.
  Capa 2 — detect_training() recovered + validate_face_quality(training_mode=True):
           rostro encontrado solo con preprocesamiento adicional, pasa filtro relajado → RECUPERADA.
  Capa 3 — sin detección válida → DESCARTADA (irrecuperable).

En todos los casos aceptados:
  - normalize_face(enhance_for_training=True): CLAHE + unsharp masking antes de HOG.
  - extract_hog_features() extrae HOG piramidal (celdas 4×4, 8×8, 16×16) para
    capturar patrones a todas las frecuencias espaciales.
  - Augmentación de escala: 5 variantes de zoom (0.5×–1.5×) por cada cara aceptada,
    simulando la misma persona a diferentes distancias de la cámara.
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
MAX_VECTORS_PER_CLASS = 400        # cap por clase: iguala clases grandes (balanceo real)

# Factores de zoom para augmentación de escala (invarianza a distancia)
# < 1.0 → simula persona lejana (cara pequeña en el frame)
# > 1.0 → simula persona muy cercana (cara grande / muy encuadrada)
_SCALE_VARIANTS = [0.50, 0.65, 0.80, 1.25, 1.50]


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de vector HOG desde una imagen y resultado de detección
# ─────────────────────────────────────────────────────────────────────────────
def _extract_vector(img: np.ndarray, face_result) -> np.ndarray | None:
    """Normaliza el rostro con enhancement y extrae el descriptor HOG piramidal.
    Devuelve None si el crop resulta inválido."""
    face_gray = normalize_face(img, face_result, enhance_for_training=True)
    if face_gray is None:
        return None
    return extract_hog_features(face_gray)


def _extract_scaled_variant(face_gray_64x64: np.ndarray, scale: float) -> np.ndarray | None:
    """Simula la misma cara a distinta distancia de la cámara.

    - scale < 1.0 (ej. 0.5): simula persona LEJOS. Toma el 50% central del crop
      (emulando que YuNet entregó un bbox pequeño), lo escala de vuelta a 64×64.
      El resultado tiene menor detalle, igual que una cara captada a mayor distancia.
    - scale > 1.0 (ej. 1.5): simula persona MUY CERCA. Amplía el centro del crop
      recortando los bordes (como si YuNet hubiera dado un bbox de cara muy grande).

    Devuelve el vector HOG piramidal de la variante, o None si no es posible.
    """
    h, w = face_gray_64x64.shape[:2]  # siempre 64×64

    if scale < 1.0:
        # Zoom-out: copiar la cara a una región central más pequeña sobre fondo gris
        inner_h = max(8, int(h * scale))
        inner_w = max(8, int(w * scale))
        canvas = np.full((h, w), 128, dtype=np.uint8)  # fondo neutro
        pad_y = (h - inner_h) // 2
        pad_x = (w - inner_w) // 2
        resized_inner = cv2.resize(face_gray_64x64, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        canvas[pad_y:pad_y + inner_h, pad_x:pad_x + inner_w] = resized_inner
        scaled = canvas
    else:
        # Zoom-in: recortar el centro ampliado de vuelta a 64×64
        crop_h = max(8, int(h / scale))
        crop_w = max(8, int(w / scale))
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - crop_h // 2)
        x1 = max(0, cx - crop_w // 2)
        y2 = min(h, y1 + crop_h)
        x2 = min(w, x1 + crop_w)
        cropped = face_gray_64x64[y1:y2, x1:x2]
        if cropped.size == 0:
            return None
        scaled = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return extract_hog_features(scaled)


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

            # Obtener el crop gris 64×64 para generar variantes de escala
            face_gray_base = normalize_face(img, face_result, enhance_for_training=True)

            class_vectors.append((vector, img))

            if face_result.was_recovered:
                recovered += 1
                logger.debug(
                    f"[RECUPERADA] {class_name}/{img_path.name} — "
                    f"conf={face_result.confidence:.3f}, sharpness={quality.sharpness:.1f}"
                )
            else:
                accepted += 1

            # ── Augmentación de escala: simula la misma cara a distinta distancia ──
            if face_gray_base is not None:
                for scale in _SCALE_VARIANTS:
                    scaled_vec = _extract_scaled_variant(face_gray_base, scale)
                    if scaled_vec is not None:
                        class_vectors.append((scaled_vec, img))

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

        # -- Cap por clase: submuestreo aleatorio si supera MAX_VECTORS_PER_CLASS --
        if len(class_vectors) > MAX_VECTORS_PER_CLASS:
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(class_vectors), size=MAX_VECTORS_PER_CLASS, replace=False)
            class_vectors = [class_vectors[i] for i in sorted(sampled_indices)]
            logger.debug(f"Clase '{class_name}': reducida a {MAX_VECTORS_PER_CLASS} vectores (cap).")

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
    logger.info(f"Dataset construido: {X.shape[0]} muestras, {len(class_names)} clases. "
                f"Dimensión vector HOG: {X.shape[1]}")

    class_weights = compute_class_weights(y)  # [REQ-ENT-01]

    # -- Pipeline: StandardScaler + PCA + LinearSVC con busqueda automatica de C --
    import warnings
    from sklearn.decomposition import PCA
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
    from sklearn.metrics import make_scorer, f1_score as sk_f1

    # PCA: preservar mas varianza para 15 clases (min entre 512 y las muestras disponibles)
    N_PCA    = min(512, X.shape[0] - 1, X.shape[1])
    N_SPLITS = 5

    logger.info(f"Dataset: {X.shape[0]} muestras x {X.shape[1]} dims")
    logger.info(f"Pipeline: StandardScaler -> PCA({N_PCA}, whiten) -> LinearSVC(balanced)")

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=N_PCA, whiten=True, random_state=42)),
        ("svm",    LinearSVC(
                       class_weight="balanced",
                       max_iter=5000,
                       dual="auto",
                       random_state=42,
                   )),
    ])

    # -- Busqueda del mejor C via GridSearchCV (3-fold interno, paralelo) --
    param_grid = {"svm__C": [0.01, 0.1, 1.0, 10.0]}
    logger.info(f"Buscando mejor C en {list(param_grid['svm__C'])} con CV 3-Fold interno...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring=make_scorer(sk_f1, average="macro", zero_division=0),
            n_jobs=-1,
            refit=False,
            verbose=0,
        )
        grid_search.fit(X, y)

    best_c = grid_search.best_params_["svm__C"]
    best_inner_f1 = grid_search.best_score_
    logger.info(f"Mejor C encontrado: {best_c} (F1-macro interno 3-fold: {best_inner_f1:.4f})")
    for params, score in zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"]):
        logger.info(f"  C={params['svm__C']}: F1-macro={score:.4f}")

    # -- Reconstruir pipeline con el mejor C y evaluar con CV 5-fold final --
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=N_PCA, whiten=True, random_state=42)),
        ("svm",    LinearSVC(
                       C=best_c,
                       class_weight="balanced",
                       max_iter=5000,
                       dual="auto",
                       random_state=42,
                   )),
    ])

    logger.info(f"Evaluacion final: {N_SPLITS}-Fold CV con C={best_c} (paralelo)...")
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(sk_f1, average="macro", zero_division=0),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        cv_results = cross_validate(
            pipeline, X, y,
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            verbose=0,
        )

    fold_accs = cv_results["test_accuracy"]
    fold_f1s  = cv_results["test_f1_macro"]
    mean_acc  = float(np.mean(fold_accs))
    mean_f1   = float(np.mean(fold_f1s))

    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s), start=1):
        logger.info(f"  Fold {i}/{N_SPLITS}: accuracy={a:.4f}, f1_macro={f:.4f}")
    logger.info(f"Resultado CV final (C={best_c}): accuracy={mean_acc:.4f}, f1_macro={mean_f1:.4f}")


    # -- Decision de guardado --
    MIN_ABSOLUTE_F1 = 0.50
    MIN_IMPROVEMENT = 0.01

    if mean_f1 < MIN_ABSOLUTE_F1:
        logger.error(
            f"F1-macro={mean_f1:.4f} esta por debajo del minimo absoluto ({MIN_ABSOLUTE_F1}). "
            "Revisa el dataset: puede haber clases con muy pocas muestras."
        )
        return

    if OUTPUT_MODEL_PATH.exists():
        old_bundle = load_pickle(OUTPUT_MODEL_PATH)
        old_f1 = old_bundle.get("cv_f1_macro", 0.0)
        improvement = mean_f1 - old_f1
        if improvement >= MIN_IMPROVEMENT:
            logger.info(f"Mejora sobre modelo existente: dF1={improvement:.4f} >= {MIN_IMPROVEMENT}. Se reemplazara.")
        else:
            logger.warning(
                f"F1 nuevo ({mean_f1:.4f}) no supera al existente ({old_f1:.4f}) "
                f"en +{MIN_IMPROVEMENT}. Se guarda igualmente."
            )

    # -- Entrenamiento final con el 100% de los datos --
    logger.info("Entrenando pipeline final con el 100% de las muestras...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        pipeline.fit(X, y)

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(
        {
            "model":       pipeline,
            "class_names": class_names,
            "cv_f1_macro": mean_f1,
            "n_pca":       N_PCA,
        },
        OUTPUT_MODEL_PATH,
    )
    logger.info(f"[OK] Pipeline SVM Facial guardado en: {OUTPUT_MODEL_PATH}")
    logger.info(f"   Clases ({len(class_names)}): {class_names}")
    logger.info(f"   F1-macro CV: {mean_f1:.4f} | Accuracy CV: {mean_acc:.4f}")


if __name__ == "__main__":
    main()
