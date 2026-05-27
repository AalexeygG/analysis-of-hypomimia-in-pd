import numpy as np


def compute_features(index_series: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """
    Возвращает 6 признаков, только на русском:
    - Медиана
    - Стандартное отклонение
    - Максимум
    - Максимальная скорость изменения
    - Доля времени выше 0,5
    - Частота пиков
    """
    x = np.asarray(index_series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "Медиана": 0.0,
            "Стандартное отклонение": 0.0,
            "Максимум": 0.0,
            "Максимальная скорость изменения": 0.0,
            "Доля времени выше 0,5": 0.0,
            "Частота пиков": 0.0,
        }

    x = np.clip(x, 0.0, 1.0)

    median_val = float(np.median(x))
    std_val = float(np.std(x, ddof=0))
    max_val = float(np.max(x))

    if x.size >= 2:
        diffs = np.abs(np.diff(x))
        max_speed = float(np.max(diffs))
    else:
        max_speed = 0.0

    frac_above = float(np.mean(x > float(threshold)))

    # локальные максимумы выше 0.2, нормированные на длину ряда
    if x.size >= 3:
        is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0.2)
        peak_rate = float(np.sum(is_peak) / x.size)
    else:
        peak_rate = 0.0

    return {
        "Медиана": median_val,
        "Стандартное отклонение": std_val,
        "Максимум": max_val,
        "Максимальная скорость изменения": max_speed,
        "Доля времени выше 0,5": frac_above,
        "Частота пиков": peak_rate,
    }
