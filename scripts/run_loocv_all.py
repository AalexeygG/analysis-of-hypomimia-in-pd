"""
Прогон LOOCV для всех тестов и всех классификаторов.
Сохраняет результаты в JSON для использования в документации.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3
from app.processing.ml import run_loocv

DB = Path(__file__).resolve().parents[1] / "data" / "bts.db"
OUT = Path(__file__).resolve().parents[1] / "docs" / "loocv_results.json"

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


all_results = {}
for test_type in ["smile", "blink"]:
    bp = get_features(test_type, "BP")
    kg = get_features(test_type, "KG")
    print(f"\n=== {test_type}: BP={len(bp)}, KG={len(kg)} ===")
    if len(bp) + len(kg) < 3:
        print("  пропуск (мало данных)")
        continue
    results = run_loocv(bp, kg)
    test_block = {"n_bp": len(bp), "n_kg": len(kg), "classifiers": []}
    for r in results:
        item = {
            "classifier": r.classifier,
            "n_samples": r.n_samples,
            "accuracy": r.accuracy,
            "f1": r.f1,
            "roc_auc": r.roc_auc if r.roc_auc == r.roc_auc else None,
            "pr_auc": r.pr_auc if r.pr_auc == r.pr_auc else None,
            "confusion": r.confusion,
            "feature_importance": r.feature_importance,
        }
        test_block["classifiers"].append(item)
        print(
            f"  {r.classifier:>18}: acc={r.accuracy:.3f}  f1={r.f1:.3f}  "
            f"roc={r.roc_auc:.3f}  pr={r.pr_auc:.3f}"
        )
    all_results[test_type] = test_block

OUT.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
print(f"\nРезультаты сохранены: {OUT}")
conn.close()
