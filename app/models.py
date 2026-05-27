from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    role = Column(String(16), nullable=False, default="operator")  # operator / admin

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    external_id = Column(String(64), unique=True, nullable=False)
    group = Column(String(8), nullable=False)  # "BP" или "KG"

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sessions = relationship("Session", back_populates="patient")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    test_type = Column(String(16), nullable=False)  # "smile" / "blink"
    status = Column(String(16), nullable=False, default="processing")  # processing/done/error
    quality_score = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    video_path = Column(Text, nullable=True)
    mean_plot_path = Column(Text, nullable=True)
    box_plot_path = Column(Text, nullable=True)
    report_path = Column(Text, nullable=True)
    csv_path = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="sessions")
    features = relationship("Feature", back_populates="session", cascade="all, delete-orphan")


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)

    name = Column(String(128), nullable=False)
    value = Column(Float, nullable=False)

    session = relationship("Session", back_populates="features")
