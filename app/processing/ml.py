"""
Машинное обучение: кросс-валидация LOOCV (каждое видео — отдельный фолд).

Классификаторы: Random Forest, SVM (RBF), Logistic Regression.
Все классификаторы используют class_weight="balanced" для компенсации дисбаланса классов.
Метрики: Accuracy, ROC-AUC, F1, confusion matrix, feature importance (RF).
"""
from __future__ import annotations

import warnings
from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

FEATURE_NAMES = [
    "Медиана",
    "Стандартное отклонение",
    "Максимум",
    "Максимальная скорость изменения",
    "Доля времени выше 0,5",
    "Частота пиков",
]


@dataclass
class LOOCVResult:
    classifier: str
    n_samples: int
    n_bp: int
    n_kg: int
    accuracy: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: list[list[int]]  # [[TN,FP],[FN,TP]] (KG=0, BP=1)
    feature_importance: dict[str, float] = field(default_factory=dict)  # only RF


def _build_matrix(
    features_list: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[f.get(name, 0.0) for name in feature_names] for f in features_list],
        dtype=float,
    )
    y = np.array(labels, dtype=int)
    return X, y


def run_loocv(
    bp_features: list[dict[str, float]],
    kg_features: list[dict[str, float]],
    feature_names: Optional[List[str]] = None,
) -> list[LOOCVResult]:
    """
    Проводит LOOCV по трём классификаторам.
    bp_features: признаки сессий группы БП (метка=1).
    kg_features: признаки сессий группы КГ (метка=0).
    Возвращает список LOOCVResult (по одному на классификатор).
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    n_bp = len(bp_features)
    n_kg = len(kg_features)
    n_total = n_bp + n_kg

    if n_total < 3:
        raise ValueError(
            f"Недостаточно данных для LOOCV: {n_total} запись(ей). Нужно минимум 3."
        )

    all_features = bp_features + kg_features
    labels = [1] * n_bp + [0] * n_kg
    X, y = _build_matrix(all_features, labels, feature_names)

    classifiers = {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
    }

    results: list[LOOCVResult] = []
    loo = LeaveOneOut()

    for clf_name, pipeline in classifiers.items():
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        y_prob_all: list[float] = []

        importances_acc: list[np.ndarray] = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Если в обучающей выборке нет обоих классов — пропускаем фолд
            if len(np.unique(y_train)) < 2:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)

            y_pred = int(pipeline.predict(X_test)[0])
            y_prob = float(pipeline.predict_proba(X_test)[0, 1])

            y_true_all.append(int(y_test[0]))
            y_pred_all.append(y_pred)
            y_prob_all.append(y_prob)

            if clf_name == "RandomForest":
                importances_acc.append(pipeline.named_steps["clf"].feature_importances_)

        if len(y_true_all) < 2:
            continue

        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        y_prob = np.array(y_prob_all)

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")

        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        imp: dict[str, float] = {}
        if importances_acc:
            mean_imp = np.mean(importances_acc, axis=0)
            imp = {name: float(v) for name, v in zip(feature_names, mean_imp)}

        results.append(LOOCVResult(
            classifier=clf_name,
            n_samples=len(y_true_all),
            n_bp=n_bp,
            n_kg=n_kg,
            accuracy=acc,
            f1=f1,
            roc_auc=auc,
            pr_auc=pr_auc,
            confusion=cm,
            feature_importance=imp,
        ))

    return results
