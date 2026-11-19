from functools import lru_cache
from pathlib import Path
from typing import Any, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Bone Age Predictor API"
    app_version: str = "1.0.0"
    debug: bool = True

    checkpoint_dir: Path = BASE_DIR / "checkpoints"
    metrics_dir: Path = BASE_DIR / "metrics"
    upload_dir: Path = BASE_DIR / "uploads"
    rf_models_dir: Path = BASE_DIR / "rf_models"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    image_size: int = 512
    device: str = "cpu"
    default_model: str = "cnn_rf"

    norm_mean: float = 0.4523
    norm_std: float = 0.2118

    @field_validator(
        "checkpoint_dir", "metrics_dir", "upload_dir", "rf_models_dir", mode="before"
    )
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        p = Path(v)
        if not p.is_absolute():
            candidate = (BASE_DIR / p).resolve()
            if candidate.exists() or not p.exists():
                return candidate
            return p.resolve()
        return p

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
