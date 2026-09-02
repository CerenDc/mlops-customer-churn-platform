"""Validated HTTP contracts matching the dbt feature mart model inputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class ChurnFeatures(BaseModel):
    """One customer row with every predictive feature used during training."""

    model_config = ConfigDict(extra="forbid")

    tenure: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float | None = Field(default=None, ge=0)
    senior_citizen: int = Field(ge=0, le=1)
    gender: Literal["Female", "Male"]
    partner: YesNo
    dependents: YesNo
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: InternetAddon
    online_backup: InternetAddon
    device_protection: InternetAddon
    tech_support: InternetAddon
    streaming_tv: InternetAddon
    streaming_movies: InternetAddon
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    has_internet: int = Field(ge=0, le=1)
    has_phone: int = Field(ge=0, le=1)
    service_count: int = Field(ge=0, le=9)
    is_month_to_month: int = Field(ge=0, le=1)
    has_long_term_contract: int = Field(ge=0, le=1)
    monthly_to_total_charge_ratio: float | None = Field(default=None, ge=0)
    tenure_group: Literal[
        "No tenure",
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49+ months",
    ]


class HealthResponse(BaseModel):
    """Liveness response independent of model availability."""

    status: Literal["ok"] = "ok"


class ModelInfoResponse(BaseModel):
    """Resolved registry identity and current loader state."""

    model_name: str
    model_version: str | None
    model_alias: str
    loading_status: Literal["not_loaded", "ready", "error"]
    error: str | None = None


class PredictionResponse(BaseModel):
    """Binary churn prediction produced by the loaded champion pipeline."""

    prediction: int
    churn: bool
    churn_probability: float
    model_name: str
    model_version: str
    model_alias: str
