"""Ejecuta K-Fold o Hold-out sobre el modelo re-entrenado antes de autorizar su promoción [REQ-ENT-05]."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class CrossValidationResult:
    mean_accuracy: float
    mean_f1_macro: float
    fold_scores: list[float]


class CrossValidationRunner:
    def __init__(self, method: str = "kfold", n_splits: int = 5, holdout_ratio: float = 0.2,
                 random_state: int = 42):
        if method not in ("kfold", "holdout"):
            raise ValueError("method debe ser 'kfold' o 'holdout'.")
        self.method = method
        self.n_splits = n_splits
        self.holdout_ratio = holdout_ratio
        self.random_state = random_state

    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray,
                 in_fold_augmenter: Any | None = None) -> CrossValidationResult:
        if self.method == "holdout":
            return self._run_holdout(model, X, y, in_fold_augmenter=in_fold_augmenter)
        return self._run_kfold(model, X, y, in_fold_augmenter=in_fold_augmenter)

    def _run_kfold(self, model: Any, X: np.ndarray, y: np.ndarray,
                   in_fold_augmenter: Any | None = None) -> CrossValidationResult:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores, f1_scores = [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            if in_fold_augmenter is not None:
                X_train, y_train = in_fold_augmenter(X_train, y_train)

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X[test_idx])

            fold_scores.append(accuracy_score(y[test_idx], preds))
            f1_scores.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))

        return CrossValidationResult(
            mean_accuracy=float(np.mean(fold_scores)),
            mean_f1_macro=float(np.mean(f1_scores)),
            fold_scores=fold_scores,
        )

    def _run_holdout(self, model: Any, X: np.ndarray, y: np.ndarray,
                     in_fold_augmenter: Any | None = None) -> CrossValidationResult:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.holdout_ratio, stratify=y, random_state=self.random_state
        )
        if in_fold_augmenter is not None:
            X_train, y_train = in_fold_augmenter(X_train, y_train)

        holdout_model = clone(model)
        holdout_model.fit(X_train, y_train)
        preds = holdout_model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        return CrossValidationResult(mean_accuracy=acc, mean_f1_macro=f1, fold_scores=[acc])