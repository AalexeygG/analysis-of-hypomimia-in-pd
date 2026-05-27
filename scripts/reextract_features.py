"""
Переэкстракция признаков для всех сессий по новой методике.
Использует существующие видео из data/videos/, обновляет таблицу features в БД.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3
from app.processing.pipeline import extract_landmarks_series, compute_index_series
from app.processing.features import compute_features
from app.config import settings

DB = Path(__file__).resolve().parents[1] / "data" / "bts.db"

conn = sqlite3.connect(DB)
cur = conn.cursor()

rows = cur.execute(
    "SELECT id, test_type, video_path FROM sessions WHERE status='done'"
).fetchall()

print(f"Найдено сессий: {len(rows)}")

ok, fail = 0, 0
for sid, test_type, video_path in rows:
    if not video_path or not Path(video_path).exists():
        print(f"[{sid}] видео не найдено: {video_path}")
        fail += 1
        continue
    try:
        pts_list, quality = extract_landmarks_series(video_path, frame_step=settings.FRAME_STEP)
        idx = compute_index_series(test_type, pts_list)
        feats = compute_features(idx, threshold=0.5)

        cur.execute("DELETE FROM features WHERE session_id = ?", (sid,))
        for name, value in feats.items():
            cur.execute(
                "INSERT INTO features (session_id, name, value) VALUES (?, ?, ?)",
                (sid, name, float(value)),
            )
        cur.execute("UPDATE sessions SET quality_score = ? WHERE id = ?", (quality, sid))
        conn.commit()
        ok += 1
        print(f"[{sid}] {test_type}  q={quality:.2f}  {', '.join(f'{k}={v:.3f}' for k,v in feats.items())}")
    except Exception as e:
        fail += 1
        print(f"[{sid}] ОШИБКА: {e}")

conn.close()
print(f"\nГотово: успешно={ok}, ошибок={fail}")
