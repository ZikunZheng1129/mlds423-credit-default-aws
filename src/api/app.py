"""FastAPI inference service for credit card default prediction."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from src.config import load_config
from src.data.feature_engineering import add_credit_features
from src.data.ingest import TARGET_COLUMN
from src.data.preprocess import clean_credit_data
from src.logging_utils import get_logger, setup_logging
from src.models.predict import load_model_artifact, load_model_metadata, predict_classes

logger = get_logger(__name__)


class CustomerRecord(BaseModel):
    """Raw credit default features accepted by the prediction endpoint."""

    LIMIT_BAL: float = Field(..., ge=0)
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int = Field(..., ge=18)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


class PredictionService:
    """Lazy model loader and prediction coordinator."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path or os.getenv("CREDIT_DEFAULT_CONFIG_PATH", "configs/config.yaml"))
        self.config = load_config(self.config_path)
        setup_logging(self.config["logging"].get("level", "INFO"))
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.model_load_error: str | None = None

    @property
    def model_path(self) -> Path:
        return Path(self.config["api"].get("model_path") or self.config["artifacts"]["best_model_path"])

    @property
    def metadata_path(self) -> Path:
        return Path(
            self.config["api"].get("metadata_path")
            or Path(self.config["artifacts"]["base_dir"]) / "best_model_metadata.json"
        )

    def load_model(self) -> None:
        """Load model and metadata once; preserve error for health reporting."""
        if self.model is not None:
            return
        try:
            self.model = load_model_artifact(self.model_path)
            self.metadata = load_model_metadata(self.metadata_path)
            self.model_load_error = None
            logger.info("Loaded model artifact from %s", self.model_path)
        except FileNotFoundError as exc:
            self.model_load_error = str(exc)
            logger.warning("Model artifact unavailable: %s", exc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.model_load_error = str(exc)
            logger.exception("Model loading failed")

    def threshold(self) -> float:
        """Resolve the prediction threshold from metadata or config."""
        metadata_threshold = self.metadata.get("threshold")
        if metadata_threshold is not None:
            return float(metadata_threshold)
        return float(self.config["api"].get("default_threshold", 0.5))

    def model_name(self) -> str | None:
        """Return the model name when metadata is available."""
        name = self.metadata.get("model_name")
        return str(name) if name else None

    def health(self) -> dict[str, Any]:
        """Return service health metadata."""
        self.load_model()
        return {
            "status": "ok",
            "model_loaded": self.model is not None,
            "model_name": self.model_name(),
            "model_artifact_path": str(self.model_path),
            "model_load_error": self.model_load_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def predict(self, records: list[CustomerRecord]) -> dict[str, Any]:
        """Run validated records through cleaning, feature engineering, and model."""
        self.load_model()
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Model artifact is not loaded. Run training first.",
                    "model_artifact_path": str(self.model_path),
                    "model_load_error": self.model_load_error,
                },
            )

        start = time.perf_counter()
        logger.info("Prediction request received with records=%s", len(records))
        raw_df = pd.DataFrame([record.model_dump() for record in records])
        raw_df[TARGET_COLUMN] = 0
        clean_df = clean_credit_data(raw_df)
        feature_df = add_credit_features(clean_df).drop(columns=["default"])
        predictions = predict_classes(self.model, feature_df, self.threshold())
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Prediction request completed records=%s latency_ms=%.3f",
            len(records),
            latency_ms,
        )

        return {
            "model_name": self.model_name() or "unknown",
            "model_version": self.metadata.get("training_timestamp"),
            "threshold": self.threshold(),
            "record_count": len(records),
            "latency_ms": round(latency_ms, 3),
            "predictions": [
                {
                    "predicted_class": int(row.predicted_class),
                    "predicted_probability": float(row.predicted_probability),
                }
                for row in predictions.itertuples(index=False)
            ],
        }


def _parse_prediction_payload(payload: dict[str, Any]) -> list[CustomerRecord]:
    """Accept a single record object or {records: [...]} wrapper."""
    raw_records = payload.get("records") if "records" in payload else payload
    if isinstance(raw_records, dict):
        raw_records = [raw_records]
    if not isinstance(raw_records, list) or not raw_records:
        raise HTTPException(status_code=422, detail="Request must be a record or {'records': [...]} object.")

    records: list[CustomerRecord] = []
    errors: list[Any] = []
    for index, raw_record in enumerate(raw_records):
        try:
            records.append(CustomerRecord.model_validate(raw_record))
        except ValidationError as exc:
            errors.append({"record_index": index, "errors": exc.errors()})
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return records


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create the FastAPI application."""
    service = PredictionService(config_path)

    @asynccontextmanager
    async def lifespan(_api: FastAPI):
        logger.info("API startup")
        service.load_model()
        yield

    api = FastAPI(
        title="Credit Card Default Prediction API",
        version=service.config["project"].get("version", "0.1.0"),
        lifespan=lifespan,
    )
    api.state.prediction_service = service
    logger.info("API application created")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @api.post("/predict")
    def predict(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            records = _parse_prediction_payload(payload)
            return service.predict(records)
        except HTTPException:
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Prediction request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return api


app = create_app()
