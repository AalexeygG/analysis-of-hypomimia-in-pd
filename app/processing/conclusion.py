from __future__ import annotations
from typing import Optional
import numpy as np


def _robust_scale_mad(x: np.ndarray) -> float:
    """
    Робастный масштаб (аналог сигмы) через MAD:
    sigma ~= 1.4826 * median(|x - median(x)|)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _safe_stats(values: list[float], min_scale: float) -> Optional[tuple[float, float]]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    med = float(np.median(arr))
    sc = _robust_scale_mad(arr)
    if not np.isfinite(sc) or sc < min_scale:
        sc = min_scale
    return med, sc


def _soft_bp_probability(d_bp: float, d_kg: float, beta: float) -> float:
    """
    P(BP) = exp(-beta*d_bp) / (exp(-beta*d_bp) + exp(-beta*d_kg))
    beta > 1 усиливает различия (меньше "залипает" на 0.5).
    """
    d_bp = float(d_bp)
    d_kg = float(d_kg)
    beta = float(beta)

    a = np.exp(-beta * d_bp)
    b = np.exp(-beta * d_kg)
    p = a / (a + b) if (a + b) > 0 else 0.5
    return float(np.clip(p, 0.0, 1.0))


def parkinson_conclusion(
    *,
    test_type: str,
    current_features: dict[str, float],
    bp_features_all: list[dict[str, float]],
    kg_features_all: list[dict[str, float]],
) -> tuple[str, float]:
    """
    Итоговая интерпретация через близость к распределениям БП и КГ (P(БП) = риск 0..1).
    Для "зажмуривания" вводим усиление beta и веса признаков, чтобы не было вечного 0.5,
    но при полном перекрытии классов честно возвращаем "неопределённо".
    """

    # Нужно хотя бы по 2 наблюдения в каждой группе
    if len(bp_features_all) < 2 or len(kg_features_all) < 2:
        return ("Недостаточно данных БП/КГ для итоговой интерпретации", 0.0)

    feature_list = [
        "Среднее значение",
        "Стандартное отклонение",
        "Максимум",
        "Максимальная скорость изменения",
        "Доля времени выше 0,5",
    ]

    # Всё в 0..1 => минимальный масштаб, чтобы не было скачков при почти одинаковой группе
    MIN_SCALE = 0.05

    # Настройки под пробу
    # Для зажмуривания обычно информативнее: Максимум, Скорость, Доля времени > 0.5
    if test_type == "blink":
        weights = {
            "Среднее значение": 0.10,
            "Стандартное отклонение": 0.10,
            "Максимум": 0.35,
            "Максимальная скорость изменения": 0.25,
            "Доля времени выше 0,5": 0.20,
        }
        beta = 3.0
        # порог неразличимости: если классы слишком близки, выводим "неопределённо"
        overlap_eps = 0.12
    else:  # smile (и остальные)
        weights = {
            "Среднее значение": 0.25,
            "Стандартное отклонение": 0.15,
            "Максимум": 0.30,
            "Максимальная скорость изменения": 0.10,
            "Доля времени выше 0,5": 0.20,
        }
        beta = 2.0
        overlap_eps = 0.10

    d_bp_num = 0.0
    d_kg_num = 0.0
    w_sum = 0.0

    for name in feature_list:
        x = current_features.get(name, None)
        if x is None:
            continue
        x = float(x)
        if not np.isfinite(x):
            continue
        x = float(np.clip(x, 0.0, 1.0))

        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue

        bp_vals = [f.get(name) for f in bp_features_all if name in f]
        kg_vals = [f.get(name) for f in kg_features_all if name in f]

        bp_stats = _safe_stats(bp_vals, MIN_SCALE)
        kg_stats = _safe_stats(kg_vals, MIN_SCALE)
        if bp_stats is None or kg_stats is None:
            continue

        med_bp, sc_bp = bp_stats
        med_kg, sc_kg = kg_stats

        # расстояния "в сигмах" от медианы (по модулю)
        d_bp = abs(x - med_bp) / sc_bp
        d_kg = abs(x - med_kg) / sc_kg

        if np.isfinite(d_bp) and np.isfinite(d_kg):
            d_bp_num += w * float(d_bp)
            d_kg_num += w * float(d_kg)
            w_sum += w

    if w_sum <= 0:
        return ("Недостаточно данных для расчёта итоговой интерпретации", 0.0)

    D_bp = d_bp_num / w_sum
    D_kg = d_kg_num / w_sum

    # Если расстояния почти одинаковые — классы перекрываются => результат неопределённый
    if abs(D_bp - D_kg) < overlap_eps:
        risk = _soft_bp_probability(D_bp, D_kg, beta)
        return ("Перекрытие классов - результат неопределённый", risk)

    risk = _soft_bp_probability(D_bp, D_kg, beta)

    if risk >= 0.70:
        return ("Выражены признаки гипомимии; признаки БП вероятны", risk)
    if risk >= 0.45:
        return ("Отмечаются отдельные признаки гипомимии; требуется дополнительная проверка", risk)
    return ("Признаки гипомимии не выражены", risk)
