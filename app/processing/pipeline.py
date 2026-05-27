from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

from app.processing.indices import mouth_width, interocular_distance, ear_both_eyes
from app.processing.normalize import smooth_ma, normalize_0_1_by_percentile, normalize_eye_open_0_1
from app.processing.features import compute_features
from app.processing.stats import mann_whitney_stats
from app.config import settings

from app.processing.plots import save_session_dynamics_plot, save_index_boxplot_single
from app.processing.report import build_csv_report, build_docx_report
from app.processing.conclusion import parkinson_conclusion


@dataclass
class ProcessResult:
    index_series: np.ndarray
    quality_score: float
    features: dict[str, float]
    stats: dict[str, dict]
    dynamics_plot_path: str
    box_plot_path: str
    report_path: str
    csv_path: str
    conclusion_text: str
    risk_score: float


def extract_landmarks_series(
    video_path: str,
    frame_step: int = 2,
) -> tuple[list[np.ndarray], float]:
    """
    frame_step=2 читает каждый 2-й кадр (по умолчанию), снижая нагрузку вдвое
    без потери диагностической информации при типичных 25-30 fps.
    frame_step=1 — обрабатывать все кадры.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видеофайл (проверь формат/кодек).")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pts_list: list[np.ndarray] = []
    total = 0
    valid = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            total += 1

            # пропускаем кадры согласно frame_step
            if frame_step > 1 and (frame_idx % frame_step) != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue

            h, w = frame.shape[:2]
            lm = res.multi_face_landmarks[0].landmark
            # 3D-координаты: x,y в пикселях; z приведён к шкале ширины кадра (MediaPipe возвращает z относительно ширины лица)
            pts = np.zeros((len(lm), 3), dtype=float)
            for i, p in enumerate(lm):
                pts[i, 0] = p.x * w
                pts[i, 1] = p.y * h
                pts[i, 2] = p.z * w

            pts_list.append(pts)
            valid += 1
    finally:
        cap.release()
        face_mesh.close()

    # quality считается от общего числа прочитанных кадров
    quality = (valid / (total / frame_step)) if total > 0 else 0.0
    return pts_list, float(min(quality, 1.0))


def compute_index_series(test_type: str, pts_per_frame: list[np.ndarray]) -> np.ndarray:
    if not pts_per_frame:
        return np.array([], dtype=float)

    if test_type == "smile":
        raw = []
        for pts in pts_per_frame:
            io = max(interocular_distance(pts), 1e-6)
            mw = mouth_width(pts)
            raw.append(mw / io)
        raw = np.array(raw, dtype=float)
        # перцентильная нормировка устойчива к тому, что пациент уже улыбается в начале
        idx = normalize_0_1_by_percentile(raw, low_pct=10.0, high_pct=95.0)
        idx = smooth_ma(idx, w=5)
        return np.clip(idx, 0.0, 1.0)

    if test_type == "blink":
        ear = []
        for pts in pts_per_frame:
            ear.append(ear_both_eyes(pts))
        ear = np.array(ear, dtype=float)
        eye_open = normalize_eye_open_0_1(ear, baseline_frames=15)
        idx = 1.0 - eye_open
        idx = smooth_ma(idx, w=5)
        return np.clip(idx, 0.0, 1.0)

    raise RuntimeError("Неизвестный тип пробы.")


def process_session(
    *,
    session_id: int,
    video_path: str,
    test_type: str,
    group: str,
    min_quality: float,
    bp_sessions_features: list[dict[str, float]],
    kg_sessions_features: list[dict[str, float]],
    plots_dir: Path,
    reports_dir: Path,
    csv_dir: Path,
) -> ProcessResult:
    pts_list, quality = extract_landmarks_series(video_path, frame_step=settings.FRAME_STEP)
    if quality < min_quality:
        raise RuntimeError(f"Недостаточное качество детекции лица: {quality:.3f}")

    idx = compute_index_series(test_type, pts_list)
    if len(idx) < 5:
        raise RuntimeError("Слишком короткий ряд индекса (недостаточно валидных кадров).")

    # 5 метрик (0..1)
    features = compute_features(idx, threshold=0.5)
    feature_names = list(features.keys())

    # Статистика БП vs КГ (для отчёта)
    bp_all = list(bp_sessions_features)
    kg_all = list(kg_sessions_features)
    if group == "BP":
        bp_all.append(features)
    else:
        kg_all.append(features)

    stats = mann_whitney_stats(bp_all, kg_all, feature_names)

    # Итог через близость к распределениям БП и КГ (из БД, без текущей записи)
    # Текущая запись ещё не добавлена в БД => bp_sessions_features / kg_sessions_features уже "чистые"
    conclusion_text, risk_score = parkinson_conclusion(
        test_type=test_type,
        current_features=features,
        bp_features_all=bp_sessions_features,
        kg_features_all=kg_sessions_features,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    dynamics_plot_path = str(plots_dir / f"sess_{session_id}_dynamics.png")
    box_plot_path = str(plots_dir / f"sess_{session_id}_index_box.png")
    report_path = str(reports_dir / f"sess_{session_id}_report.docx")
    csv_path = str(csv_dir / f"sess_{session_id}_features.csv")

    save_session_dynamics_plot(idx, dynamics_plot_path, test_type, session_id=session_id)
    save_index_boxplot_single(idx, box_plot_path, test_type)

    build_docx_report(
        report_path,
        features,
        stats,
        test_type,
        group,
        quality,
        dynamics_plot_path=dynamics_plot_path,
        box_plot_path=box_plot_path,
        conclusion_text=conclusion_text,
        risk_score=risk_score,
    )
    build_csv_report(
        csv_path,
        features,
        stats,
        test_type,
        group,
        quality,
        conclusion_text=conclusion_text,
        risk_score=risk_score,
    )

    return ProcessResult(
        index_series=idx,
        quality_score=quality,
        features=features,
        stats=stats,
        dynamics_plot_path=dynamics_plot_path,
        box_plot_path=box_plot_path,
        report_path=report_path,
        csv_path=csv_path,
        conclusion_text=conclusion_text,
        risk_score=risk_score,
    )
