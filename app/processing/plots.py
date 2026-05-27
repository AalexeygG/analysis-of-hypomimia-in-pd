from __future__ import annotations
from typing import Optional
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


def save_session_dynamics_plot(
    idx: np.ndarray,
    out_path: str,
    test_type: str,
    session_id: Optional[int] = None,
):
    """
    Динамика индекса конкретной видеозаписи (одна линия, 0..1).
    """
    title_map = {
        "smile": "Динамика индекса улыбки (данная запись)",
        "blink": "Динамика индекса зажмуривания (данная запись)",
    }

    idx = np.asarray(idx, dtype=float)
    idx = np.clip(idx, 0.0, 1.0)

    plt.figure(figsize=(8, 4))
    plt.plot(idx)
    title = title_map.get(test_type, "Динамика индекса (данная запись)")
    if session_id is not None:
        title += f" — сессия {session_id}"
    plt.title(title)
    plt.xlabel("Кадр")
    plt.ylabel("Индекс (0…1)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_index_boxplot_single(
    idx: np.ndarray,
    out_path: str,
    test_type: str,
):
    """
    Boxplot индекса для ОДНОЙ видеозаписи: один столбец, диапазон 0..1.
    """
    title_map = {
        "smile": "Boxplot индекса улыбки (данная запись)",
        "blink": "Boxplot индекса зажмуривания (данная запись)",
    }

    x = np.asarray(idx, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, 1.0)

    plt.figure(figsize=(5.2, 4.2))
    if x.size == 0:
        plt.title(title_map.get(test_type, "Boxplot индекса (данная запись)"))
        plt.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()
        return

    plt.boxplot([x], widths=0.5, showmeans=True)
    plt.title(title_map.get(test_type, "Boxplot индекса (данная запись)"))
    plt.xticks([1], ["Индекс"])
    plt.ylabel("Индекс (0…1)")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
