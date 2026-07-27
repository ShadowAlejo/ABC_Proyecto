"""Compila todas las métricas y gráficas en un reporte consolidado (PDF/HTML/JSON) por ejecución [REQ-EVL-01]."""
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics_calculator import ClassMetrics
from evaluation.roc_curve_generator import ROCResult
from evaluation.cmc_curve_generator import CMCResult
from utils.file_io_helpers import ensure_dir


class EvaluationReportWriter:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)

    def write_report(self, metrics_per_class: Dict[str, ClassMetrics], confusion_matrix: np.ndarray,
                      class_names: List[str], roc_result: ROCResult, cmc_result: CMCResult) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(self.output_dir / f"eval_{timestamp}")

        json_report = {
            "timestamp": timestamp,
            "metrics_per_class": {
                name: {"recall": m.recall, "precision": m.precision, "specificity": m.specificity}
                for name, m in metrics_per_class.items()
            },
            "roc_auc": roc_result.auc_score,
            "optimal_threshold": roc_result.optimal_threshold,
            "cmc_rank1": cmc_result.rank1,
            "cmc_rank5": cmc_result.rank5,
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)

        self._plot_confusion_matrix(confusion_matrix, class_names, run_dir / "confusion_matrix.png")
        self._plot_roc_curve(roc_result, run_dir / "roc_curve.png")
        self._plot_cmc_curve(cmc_result, run_dir / "cmc_curve.png")

        self._write_html_summary(run_dir, json_report)
        return run_dir

    @staticmethod
    def _plot_confusion_matrix(matrix: np.ndarray, class_names: List[str], output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión 16x16 + Desconocido")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_roc_curve(roc_result: ROCResult, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(roc_result.fpr, roc_result.tpr, label=f"AUC = {roc_result.auc_score:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        ax.set_title("Curva ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_cmc_curve(cmc_result: CMCResult, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(cmc_result.ranks, cmc_result.accuracies, marker="o")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Exactitud Acumulada")
        ax.set_title(f"Curva CMC (Rank-1={cmc_result.rank1:.3f}, Rank-5={cmc_result.rank5:.3f})")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _write_html_summary(run_dir: Path, report: dict) -> None:
        html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Reporte de Evaluación</title></head>
<body>
<h1>Reporte de Evaluación ID/Re-ID — {report['timestamp']}</h1>
<p><b>AUC ROC:</b> {report['roc_auc']:.4f} | <b>Umbral óptimo:</b> {report['optimal_threshold']:.4f}</p>
<p><b>Rank-1:</b> {report['cmc_rank1']:.4f} | <b>Rank-5:</b> {report['cmc_rank5']:.4f}</p>
<img src="confusion_matrix.png" width="600"><br>
<img src="roc_curve.png" width="400"><img src="cmc_curve.png" width="400">
</body></html>"""
        with open(run_dir / "report.html", "w", encoding="utf-8") as f:
            f.write(html)