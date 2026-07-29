"""Entrena el SVM Facial (15 clases) a partir de dataset/raw_images/<sujeto>/*.jpg.

Estrategia de extraccion de muestras (maximo recall):
  Capa 1 - detect_training() + validate_face_quality(training_mode=True):
           imagen limpia que pasa los filtros relajados -> ACEPTADA.
  Capa 2 - detect_training() recovered + validate_face_quality(training_mode=True):
           rostro encontrado solo con preprocesamiento adicional -> RECUPERADA.
  Capa 3 - sin deteccion valida -> DESCARTADA.

Pipeline de features (identico al usado en inferencia):
  normalize_face(): crop -> alineacion por landmarks -> CLAHE -> unsharp -> 64x64
  extract_combined_features(): HOG piramidal (5814) + LBP uniforme grid 4x4 (160) = 5974 dims

Augmentacion sistematica (solo ESCALA y ROTACIONES moderadas):
  - Escala: 5 variantes de zoom [0.5x-1.5x] -> invarianza a distancia
  - Rotacion: +-7 y +-15 grados -> inclinaciones reales de cabeza
  - Flip DESACTIVADO: la asimetria facial es informacion discriminativa valiosa
"""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))  # permite importar módulos del proyecto

from classification.svm_facial_model import SVMFacialModel
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
MAX_VECTORS_PER_CLASS = 700        # cap por clase: iguala clases grandes (balanceo real)

# Factores de zoom para augmentación de escala (invarianza a distancia)
# < 1.0 → simula persona lejana (cara pequeña en el frame)
# > 1.0 → simula persona muy cercana (cara grande / muy encuadrada)
_SCALE_VARIANTS = [0.50, 0.65, 0.80, 1.25, 1.50]

# Rotaciones en grados para invarianza a inclinacion de cabeza.
# Rango reducido a +-15 grados: inclinaciones mayores no ocurren en produccion
# y aumentan la varianza intra-clase sin beneficio real.
_ROTATION_VARIANTS = [-15, -7, 7, 15]

# Flip horizontal DESACTIVADO: la asimetria facial (posicion de lunares, cicatrices,
# forma asimetrica de ojos) es informacion discriminativa clave para identificacion.
# Activar el flip destruye esta informacion al ensenar que cara y espejo son la misma persona.
_APPLY_FLIP = False


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de vector HOG desde una imagen y resultado de detección
# ─────────────────────────────────────────────────────────────────────────────
def _extract_vector(img: np.ndarray, face_result) -> np.ndarray | None:
    """Normaliza el rostro (crop+align+CLAHE) y extrae el descriptor 100% HOG.
    Devuelve None si el crop resulta invalido."""
    face_gray, landmarks = normalize_face(img, face_result)
    if face_gray is None:
        return None
    return extract_hog_features(face_gray, landmarks)


def _extract_scaled_variant(face_gray_64x64: np.ndarray, landmarks: list, scale: float) -> np.ndarray | None:
    """Simula la misma cara a distinta distancia de la cámara y escala los landmarks."""
    h, w = face_gray_64x64.shape[:2]

    if scale < 1.0:
        inner_h = max(8, int(h * scale))
        inner_w = max(8, int(w * scale))
        canvas = np.full((h, w), 128, dtype=np.uint8)
        pad_y = (h - inner_h) // 2
        pad_x = (w - inner_w) // 2
        resized_inner = cv2.resize(face_gray_64x64, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        canvas[pad_y:pad_y + inner_h, pad_x:pad_x + inner_w] = resized_inner
        scaled = canvas
    else:
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

    scaled_landmarks = []
    if landmarks:
        scaled_landmarks = [((lx-32)*scale + 32, (ly-32)*scale + 32) for lx, ly in landmarks]

    return extract_hog_features(scaled, scaled_landmarks)


def _extract_rotated_variant(face_gray_64x64: np.ndarray, landmarks: list, angle: float) -> np.ndarray | None:
    """Simula inclinación de la cabeza rotando la imagen y los landmarks."""
    h, w = face_gray_64x64.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(face_gray_64x64, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    rotated_landmarks = []
    if landmarks:
        for lx, ly in landmarks:
            nx = matrix[0,0]*lx + matrix[0,1]*ly + matrix[0,2]
            ny = matrix[1,0]*lx + matrix[1,1]*ly + matrix[1,2]
            rotated_landmarks.append((nx, ny))
            
    return extract_hog_features(rotated, rotated_landmarks)


def _extract_flipped_variant(face_gray_64x64: np.ndarray, landmarks: list) -> np.ndarray | None:
    """Invertir imagen horizontalmente e intercambiar posición de ojos."""
    flipped = cv2.flip(face_gray_64x64, 1)
    flipped_landmarks = []
    if landmarks:
        for lx, ly in landmarks:
            flipped_landmarks.append((64.0 - lx, ly))
        if len(flipped_landmarks) >= 5:
            # Intercambiar Ojo Izq (0) con Ojo Der (1) y Boca Izq (3) con Boca Der (4)
            flipped_landmarks[0], flipped_landmarks[1] = flipped_landmarks[1], flipped_landmarks[0]
            flipped_landmarks[3], flipped_landmarks[4] = flipped_landmarks[4], flipped_landmarks[3]
            
    return extract_hog_features(flipped, flipped_landmarks)


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

            # Obtener el crop gris 64×64 y sus landmarks para generar variantes de augmentacion
            face_gray_base, landmarks_base = normalize_face(img, face_result, enhance_for_training=True)

            class_vectors.append((vector, img))

            if face_result.was_recovered:
                recovered += 1
                logger.debug(
                    f"[RECUPERADA] {class_name}/{img_path.name} — "
                    f"conf={face_result.confidence:.3f}, sharpness={quality.sharpness:.1f}"
                )
            else:
                accepted += 1

            # ── Augmentación sistemática: Escala, Rotación, Flip ──
            if face_gray_base is not None:
                # 1. Escalas (distancia)
                for scale in _SCALE_VARIANTS:
                    scaled_vec = _extract_scaled_variant(face_gray_base, landmarks_base, scale)
                    if scaled_vec is not None:
                        class_vectors.append((scaled_vec, img))
                
                # 2. Rotaciones (inclinación de cabeza)
                for angle in _ROTATION_VARIANTS:
                    rot_vec = _extract_rotated_variant(face_gray_base, landmarks_base, angle)
                    if rot_vec is not None:
                        class_vectors.append((rot_vec, img))
                
                # 3. Flip horizontal (perfiles / simetría)
                if _APPLY_FLIP:
                    flip_vec = _extract_flipped_variant(face_gray_base, landmarks_base)
                    if flip_vec is not None:
                        class_vectors.append((flip_vec, img))

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
            f"[OK] aceptadas={accepted}  "
            f"[REC] recuperadas={recovered}  "
            f"[ERR] descartadas={discarded}  "
            f"-> {len(class_vectors)} vectores"
        )

    # Resumen global
    logger.info("-" * 60)
    logger.info("RESUMEN GLOBAL DEL DATASET:")
    logger.info(f"  [OK] Aceptadas (directas)  : {total_accepted}")
    logger.info(f"  [REC] Recuperadas (preproc.) : {total_recovered}")
    logger.info(f"  [ERR] Descartadas            : {total_discarded}")
    logger.info(
        f"  Tasa de cobertura: "
        f"{(total_accepted + total_recovered) / max(1, total_accepted + total_recovered + total_discarded) * 100:.1f}%"
    )
    logger.info("-" * 60)

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
    MIN_ABSOLUTE_F1 = 0.30
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
