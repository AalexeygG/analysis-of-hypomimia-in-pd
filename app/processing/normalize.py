import numpy as np


def smooth_ma(x: np.ndarray, w: int = 5) -> np.ndarray:
    """Скользящее среднее (сглаживание)."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="same")


def normalize_0_1_by_baseline(x: np.ndarray, baseline_frames: int = 15) -> np.ndarray:
    """
    Приведение ряда к диапазону [0;1] относительно нейтрального уровня (baseline)
    и устойчивого максимума (95-й перцентиль).
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x

    n0 = min(baseline_frames, len(x))
    baseline = float(np.median(x[:n0]))
    top = float(np.percentile(x, 95))
    denom = max(top - baseline, 1e-9)

    y = (x - baseline) / denom
    return np.clip(y, 0.0, 1.0)


def normalize_0_1_by_percentile(x: np.ndarray, low_pct: float = 10.0, high_pct: float = 95.0) -> np.ndarray:
    """
    Нормировка по перцентилям всего ряда. Устойчива к тому, что пациент уже улыбается
    в первых кадрах: baseline берётся как минимально-«нейтральное» состояние во всём видео.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x

    baseline = float(np.percentile(x, low_pct))
    top = float(np.percentile(x, high_pct))
    denom = max(top - baseline, 1e-9)

    y = (x - baseline) / denom
    return np.clip(y, 0.0, 1.0)


def normalize_eye_open_0_1(ear: np.ndarray, baseline_frames: int = 15) -> np.ndarray:
    """
    EAR -> степень открытости глаза в [0;1], где 1 — открыт, 0 — закрыт.
    """
    ear = np.asarray(ear, dtype=float)
    if len(ear) == 0:
        return ear

    n0 = min(baseline_frames, len(ear))
    open_level = float(np.percentile(ear[:n0], 90))
    closed_level = float(np.percentile(ear, 10))
    denom = max(open_level - closed_level, 1e-9)

    y = (ear - closed_level) / denom
    return np.clip(y, 0.0, 1.0)
