from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str = "postgresql+psycopg2://bts_user:bts_pass@127.0.0.1:5432/bts_db"
    DATA_DIR: str = str(Path(__file__).resolve().parents[1] / "data")

    # Auth
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 часов

    # качество: минимальная доля кадров с лицом
    MIN_QUALITY_SCORE: float = 0.5

    # нормирование времени для "средней динамики" (сколько точек на оси X)
    MEAN_DYNAMICS_POINTS: int = 120

    # шаг по кадрам при обработке видео (1 = все кадры, 2 = каждый второй и т.д.)
    FRAME_STEP: int = 2


settings = Settings()
