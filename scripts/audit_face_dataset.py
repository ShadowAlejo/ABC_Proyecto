"""Audita dataset/raw_images/ detectando rostros con la estrategia robusta y reporta
casos problemáticos (falsos negativos y candidatos ambiguos) antes de entrenar."""
import sys
from pathlib import Path
import csv
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_extraction.face.yunet_face_detector import YuNetFaceDetector
from feature_extraction.face.face_quality_validator import validate_face_quality
from utils.file_io_helpers import list_files, ensure_dir
from utils.logger import get_logger

logger = get_logger("audit_face_dataset")

RAW_IMAGES_DIR = Path("dataset/raw_images")
REPORT_DIR = Path("reports/dataset_audit")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    detector = YuNetFaceDetector()
    ensure_dir(REPORT_DIR)

    rows = []
    total, no_face_count, ambiguous_count, low_quality_count = 0, 0, 0, 0

    for class_dir in sorted(RAW_IMAGES_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in list_files(class_dir, IMAGE_EXTENSIONS):
            total += 1
            img = cv2.imread(str(img_path))
            if img is None:
                rows.append([class_dir.name, img_path.name, "ERROR_LECTURA", "", ""])
                continue

            result = detector.detect(img)
            n_candidates = len(result.all_candidates)
            status = "OK"

            if not result.detected:
                status = "FALSO_NEGATIVO_SIN_ROSTRO"
                no_face_count += 1
            elif n_candidates > 1:
                status = "AMBIGUO_MULTIPLES_CANDIDATOS"
                ambiguous_count += 1
            else:
                quality = validate_face_quality(img, result)
                if not quality.is_valid:
                    status = f"BAJA_CALIDAD:{'|'.join(quality.reasons)}"
                    low_quality_count += 1

            rows.append([class_dir.name, img_path.name, status,
                         f"{result.confidence:.3f}", str(n_candidates)])

    report_path = REPORT_DIR / "face_audit_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sujeto", "archivo", "estado", "confianza", "num_candidatos"])
        writer.writerows(rows)

    logger.info(f"Auditoría completa: {total} imágenes procesadas.")
    logger.info(f"  Falsos negativos (sin rostro): {no_face_count}")
    logger.info(f"  Ambiguos (múltiples candidatos): {ambiguous_count}")
    logger.info(f"  Baja calidad (borroso/pequeño/perfil): {low_quality_count}")
    logger.info(f"Reporte detallado en: {report_path}")

    if no_face_count > 0:
        logger.warning(
            "Revisa manualmente las imágenes marcadas FALSO_NEGATIVO_SIN_ROSTRO: "
            "puede que necesiten mejor iluminación, recorte, o simplemente deban eliminarse."
        )


if __name__ == "__main__":
    main()