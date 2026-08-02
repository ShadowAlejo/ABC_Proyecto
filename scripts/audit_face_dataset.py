"""Audita dataset/raw_images/ comprobando resolución, brillo y nitidez de cada imagen sin YuNet."""
import sys
from pathlib import Path
import csv
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.face_quality_validator import validate_face_quality
from utils.file_io_helpers import list_files, ensure_dir, load_image
from utils.logger import get_logger

logger = get_logger("audit_face_dataset")

RAW_IMAGES_DIR = Path("dataset/raw_images")
REPORT_DIR = Path("reports/dataset_audit")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    ensure_dir(REPORT_DIR)

    rows = []
    total, low_quality_count = 0, 0

    for class_dir in sorted(RAW_IMAGES_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in list_files(class_dir, IMAGE_EXTENSIONS):
            total += 1
            img = load_image(img_path)
            if img is None:
                rows.append([class_dir.name, img_path.name, "ERROR_LECTURA", "0.0", "0.0"])
                continue

            head_crop = normalize_face(img, is_body_roi=False)
            if head_crop is None:
                status = "RECORTE_INVALIDO"
                low_quality_count += 1
                rows.append([class_dir.name, img_path.name, status, "0.0", "0.0"])
                continue

            quality = validate_face_quality(head_crop, training_mode=True)
            if not quality.is_valid:
                status = f"BAJA_CALIDAD:{'|'.join(quality.reasons)}"
                low_quality_count += 1
            else:
                status = "OK"

            rows.append([class_dir.name, img_path.name, status, f"{quality.sharpness:.1f}", f"{quality.brightness:.1f}"])

    report_path = REPORT_DIR / "face_audit_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sujeto", "archivo", "estado", "nitidez", "brillo"])
        writer.writerows(rows)

    logger.info(f"Auditoría completa: {total} imágenes procesadas.")
    logger.info(f"  Baja calidad (borroso/pequeño/oscuro): {low_quality_count}")
    logger.info(f"Reporte detallado en: {report_path}")


if __name__ == "__main__":
    main()