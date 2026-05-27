import numpy as np

# Индексы FaceMesh (стандартные)
MOUTH_L = 61
MOUTH_R = 291

EYE_L_OUT = 33
EYE_R_OUT = 263

# Наборы точек для EAR (Eye Aspect Ratio)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def interocular_distance(pts: np.ndarray) -> float:
    # расстояние между внешними уголками глаз (масштаб лица); 3D — устойчиво к повороту головы
    return _dist(pts[EYE_L_OUT], pts[EYE_R_OUT])


def mouth_width(pts: np.ndarray) -> float:
    return _dist(pts[MOUTH_L], pts[MOUTH_R])


def ear_for_eye(pts: np.ndarray, eye_idx: list[int]) -> float:
    # EAR: (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||) — расстояния в 3D
    p1, p2, p3, p4, p5, p6 = eye_idx
    A = _dist(pts[p2], pts[p6])
    B = _dist(pts[p3], pts[p5])
    C = max(_dist(pts[p1], pts[p4]), 1e-9)
    return float((A + B) / (2.0 * C))


def ear_both_eyes(pts: np.ndarray) -> float:
    return float((ear_for_eye(pts, LEFT_EYE) + ear_for_eye(pts, RIGHT_EYE)) / 2.0)
