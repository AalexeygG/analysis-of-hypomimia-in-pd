from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from app.models import Patient, Session as Sess, Feature


def get_patient_by_external_id(db: Session, external_id: str) -> Optional[Patient]:
    return db.execute(select(Patient).where(Patient.external_id == external_id)).scalar_one_or_none()


def create_patient(db: Session, external_id: str, group: str) -> Patient:
    p = Patient(external_id=external_id, group=group)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def create_session(db: Session, patient_id: int, test_type: str, video_path: str) -> Sess:
    s = Sess(patient_id=patient_id, test_type=test_type, status="processing", video_path=video_path)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def set_session_done(
    db: Session,
    session_id: int,
    quality_score: float,
    mean_plot_path: str,
    box_plot_path: str,
    report_path: str,
    csv_path: str,
):
    s = db.get(Sess, session_id)
    s.status = "done"
    s.quality_score = float(quality_score)
    s.mean_plot_path = mean_plot_path
    s.box_plot_path = box_plot_path
    s.report_path = report_path
    s.csv_path = csv_path
    db.commit()


def set_session_error(db: Session, session_id: int, message: str):
    s = db.get(Sess, session_id)
    s.status = "error"
    s.error_message = message
    db.commit()


def replace_features(db: Session, session_id: int, features: dict[str, float]):
    db.execute(delete(Feature).where(Feature.session_id == session_id))
    db.commit()

    for k, v in features.items():
        db.add(Feature(session_id=session_id, name=k, value=float(v)))
    db.commit()


def get_session(db: Session, session_id: int) -> Optional[Sess]:
    return db.get(Sess, session_id)


def get_sessions_done_by_test_and_group(db: Session, test_type: str, group: str) -> list[Sess]:
    q = (
        select(Sess)
        .join(Patient, Patient.id == Sess.patient_id)
        .where(Sess.status == "done")
        .where(Sess.test_type == test_type)
        .where(Patient.group == group)
        .order_by(Sess.id.asc())
    )
    return list(db.execute(q).scalars().all())


def get_features_for_session(db: Session, session_id: int) -> dict[str, float]:
    q = select(Feature).where(Feature.session_id == session_id)
    rows = db.execute(q).scalars().all()
    return {r.name: float(r.value) for r in rows}
