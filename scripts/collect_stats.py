"""Сбор Mann-Whitney и сводных таблиц признаков для документации."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3
import numpy as np
from scipy.stats import mannwhitneyu

DB = Path(__file__).resolve().parents[1] / "data" / "bts.db"
OUT = Path(__file__).resolve().parents[1] / "docs" / "stats.json"

FEATURE_NAMES = [
    "Медиана",
    "Стандартное отклонение",
    "Максимум",
    "Максимальная скорость изменения",
    "Доля времени выше 0,5",
    "Частота пиков",
]

conn = sqlite3.connect(DB)
cur = conn.cursor()


def get_features(test_type: str, group: str):
    sess_ids = cur.execute(
        """
        SELECT s.id FROM sessions s
        JOIN patients p ON p.id = s.patient_id
        WHERE s.status='done' AND s.test_type=? AND p."group"=?
        """,
        (test_type, group),
    ).fetchall()
    out = []
    for (sid,) in sess_ids:
        rows = cur.execute(
            "SELECT name, value FROM features WHERE session_id=?", (sid,)
        ).fetchall()
        if rows:
            out.append({name: value for name, value in rows})
    return out


all_stats = {}
for test_type in ["smile", "blink"]:
    bp = get_features(test_type, "BP")
    kg = get_features(test_type, "KG")
    if len(bp) < 2 or len(kg) < 2:
        continue
    block = {"n_bp": len(bp), "n_kg": len(kg), "features": []}
    for name in FEATURE_NAMES:
        bp_vals = np.array([f.get(name, 0.0) for f in bp])
        kg_vals = np.array([f.get(name, 0.0) for f in kg])
        try:
            U, p = mannwhitneyu(bp_vals, kg_vals, alternative="two-sided")
            U, p = float(U), float(p)
        except ValueError:
            U, p = float("nan"), float("nan")
        block["features"].append({
            "name": name,
            "bp_mean": float(bp_vals.mean()),
            "bp_std": float(bp_vals.std(ddof=0)),
            "bp_median": float(np.median(bp_vals)),
            "kg_mean": float(kg_vals.mean()),
            "kg_std": float(kg_vals.std(ddof=0)),
            "kg_median": float(np.median(kg_vals)),
            "U": U,
            "p_value": p,
        })
    all_stats[test_type] = block

OUT.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2))
print(f"Сохранено: {OUT}")
for tt, block in all_stats.items():
    print(f"\n{tt}: BP={block['n_bp']}, KG={block['n_kg']}")
    for f in block["features"]:
        p = f["p_value"]
        ps = f"{p:.4f}" if p == p else "nan"
        print(f"  {f['name']:35s} BP={f['bp_mean']:.3f}±{f['bp_std']:.3f}  KG={f['kg_mean']:.3f}±{f['kg_std']:.3f}  p={ps}")
conn.close()
