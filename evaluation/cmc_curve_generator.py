"""Genera curvas CMC para medir exactitud Rank-1 y Rank-5 del sistema de identificación [REQ-EVL-01]."""
from dataclasses import dataclass
import numpy as np


@dataclass
class CMCResult:
    ranks: np.ndarray
    accuracies: np.ndarray
    rank1: float
    rank5: float


def generate_cmc_curve(sorted_prediction_ranks: np.ndarray, max_rank: int = 10) -> CMCResult:
    """
    sorted_prediction_ranks: array donde cada elemento indica en qué posición (rank, 1-indexado)
    apareció la clase verdadera dentro del ranking de probabilidades del SVM para esa muestra.
    """
    n_samples = len(sorted_prediction_ranks)
    ranks = np.arange(1, max_rank + 1)
    accuracies = np.array([
        np.mean(sorted_prediction_ranks <= r) for r in ranks
    ], dtype=np.float32)

    rank1 = float(accuracies[0]) if max_rank >= 1 else 0.0
    rank5 = float(accuracies[4]) if max_rank >= 5 else float(accuracies[-1])

    return CMCResult(ranks=ranks, accuracies=accuracies, rank1=rank1, rank5=rank5)