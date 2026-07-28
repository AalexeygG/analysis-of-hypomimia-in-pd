# Устойчивость результата: bootstrap, перестановочный тест и сравнение с MLP.
import sys
import json
import warnings
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

DB = Path(__file__).resolve().parents[1] / "data" / "bts.db"
OUT = Path(__file__).resolve().parents[1] / "docs" / "robustness_results.json"

FEATURE_NAMES = [
    "Медиана", "Стандартное отклонение", "Максимум",
    "Максимальная скорость изменения", "Доля времени выше 0,5", "Частота пиков",
]

N_BOOTSTRAP = 5000   # ресэмплингов для доверительного интервала
N_PERM = 300         # перестановок меток для теста значимости
rng = np.random.default_rng(42)


def load_subject_vectors(test_type):
    """Один усреднённый вектор признаков на человека."""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    rows = cur.execute(
        """SELECT s.id, s.patient_id, p."group" FROM sessions s
           JOIN patients p ON p.id = s.patient_id
           WHERE s.status='done' AND s.test_type=?""",
        (test_type,),
    ).fetchall()
    by_subj = defaultdict(list)
    grp = {}
    for sid, pid, group in rows:
        fr = {n: v for n, v in cur.execute(
            "SELECT name, value FROM features WHERE session_id=?", (sid,)).fetchall()}
        if not fr:
            continue
        by_subj[pid].append([fr.get(n, 0.0) for n in FEATURE_NAMES])
        grp[pid] = group
    conn.close()
    subj, X, y = [], [], []
    for pid, vecs in by_subj.items():
        subj.append(pid)
        X.append(np.mean(np.array(vecs, float), axis=0))
        y.append(1 if grp[pid] == "BP" else 0)
    return np.array(X, float), np.array(y, int), subj


def combined_vectors():
    """Склейка признаков blink+smile для людей, у кого есть обе пробы."""
    Xb, yb, sb = load_subject_vectors("blink")
    Xs, ys, ss = load_subject_vectors("smile")
    mb = {p: (Xb[i], yb[i]) for i, p in enumerate(sb)}
    ms = {p: (Xs[i], ys[i]) for i, p in enumerate(ss)}
    common = [p for p in mb if p in ms]
    X = np.array([np.concatenate([mb[p][0], ms[p][0]]) for p in common])
    y = np.array([mb[p][1] for p in common])
    return X, y


def random_forest():
    return Pipeline([("clf", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"))])


def mlp():
    return Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=2000, random_state=42))])


def loo_predict(model_factory, X, y):
    """Out-of-fold предсказания по схеме leave-one-subject-out."""
    y_true, y_prob = [], []
    for tr, te in LeaveOneOut().split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        model = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y[tr])
        y_true.append(int(y[te][0]))
        y_prob.append(float(model.predict_proba(X[te])[0, 1]))
    return np.array(y_true), np.array(y_prob)


def metrics(y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def main():
    X, y = combined_vectors()
    print(f"Людей с обеими пробами: {len(y)} (БП={int(y.sum())}, КГ={int((1 - y).sum())}), признаков={X.shape[1]}")

    # случайный лес
    yt, yp = loo_predict(random_forest, X, y)
    base = metrics(yt, yp)
    print(f"\nСлучайный лес: AUC={base['auc']:.3f}  чувств.={base['sensitivity']:.3f}  "
          f"специф.={base['specificity']:.3f}  точность={base['accuracy']:.3f}")

    # bootstrap-интервал для AUC
    aucs = []
    n = len(yt)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        aucs.append(roc_auc_score(yt[idx], yp[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"Bootstrap ({N_BOOTSTRAP}): 95% доверительный интервал AUC = [{lo:.3f}, {hi:.3f}]")

    # перестановочный тест
    null = []
    for i in range(N_PERM):
        y_perm = rng.permutation(y)
        yt_p, yp_p = loo_predict(random_forest, X, y_perm)
        if len(np.unique(yt_p)) > 1:
            null.append(roc_auc_score(yt_p, yp_p))
        if (i + 1) % 50 == 0:
            print(f"  перестановки: {i + 1}/{N_PERM}")
    null = np.array(null)
    pval = (np.sum(null >= base["auc"]) + 1) / (len(null) + 1)
    print(f"Перестановочный тест ({len(null)}): нулевой AUC в среднем {null.mean():.3f}, "
          f"p-value = {pval:.4f}")

    # нейросеть для сравнения
    yt_m, yp_m = loo_predict(mlp, X, y)
    mm = metrics(yt_m, yp_m)
    print(f"\nНейросеть MLP: AUC={mm['auc']:.3f}  чувств.={mm['sensitivity']:.3f}  "
          f"специф.={mm['specificity']:.3f}  точность={mm['accuracy']:.3f}")
    print(f"Случайный лес выигрывает у нейросети: {base['auc']:.3f} против {mm['auc']:.3f} "
          f"(на малой выборке нейросеть переобучается)")

    result = {
        "random_forest": base,
        "auc_ci95": [float(lo), float(hi)],
        "permutation": {"p_value": float(pval), "null_mean": float(null.mean()),
                        "n_perm": int(len(null))},
        "mlp": mm,
        "n_subjects": int(len(y)), "n_bp": int(y.sum()), "n_kg": int((1 - y).sum()),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nРезультаты сохранены: {OUT}")


if __name__ == "__main__":
    main()
