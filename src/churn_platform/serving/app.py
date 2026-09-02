"""FastAPI application exposing champion model inference and metrics."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from churn_platform.serving.model_loader import (
    ChampionModelLoader,
    ModelUnavailableError,
)
from churn_platform.serving.schemas import (
    ChurnFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

HTTP_REQUESTS = Counter(
    "churn_serving_http_requests_total",
    "HTTP requests handled by the prediction API.",
    ("method", "path", "status"),
)
INFERENCE_REQUESTS = Counter(
    "churn_serving_inference_requests_total",
    "Valid inference requests handled by the prediction API.",
)
INFERENCE_FAILURES = Counter(
    "churn_serving_inference_failures_total",
    "Inference requests that failed after validation.",
)
PREDICTIONS = Counter(
    "churn_serving_predictions_total",
    "Predictions produced by class.",
    ("prediction",),
)
INFERENCE_LATENCY = Histogram(
    "churn_serving_inference_duration_seconds",
    "Champion model prediction latency.",
)


def configured_loader(request: Request) -> ChampionModelLoader:
    """Return the application-scoped lazy model loader."""
    return request.app.state.model_loader


LoaderDependency = Annotated[ChampionModelLoader, Depends(configured_loader)]


def create_app(loader: ChampionModelLoader | None = None) -> FastAPI:
    """Build an application whose loader can be replaced in isolated tests."""
    application = FastAPI(
        title="Customer Churn Prediction API",
        version="13.0.0",
        description="Inference from the MLflow Model Registry champion pipeline.",
    )
    application.state.model_loader = loader or ChampionModelLoader()

    @application.middleware("http")
    async def observe_http(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        ).inc()
        return response

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @application.get("/model/info", response_model=ModelInfoResponse)
    def model_info(
        model_loader: LoaderDependency,
    ) -> ModelInfoResponse:
        return ModelInfoResponse.model_validate(model_loader.info())

    @application.get("/ready", response_model=HealthResponse, include_in_schema=False)
    def readiness(model_loader: LoaderDependency) -> HealthResponse:
        try:
            model_loader.load()
        except ModelUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        return HealthResponse()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        features: ChurnFeatures,
        model_loader: LoaderDependency,
    ) -> PredictionResponse:
        INFERENCE_REQUESTS.inc()
        started = perf_counter()
        try:
            result = model_loader.predict(features)
        except ModelUnavailableError as error:
            INFERENCE_FAILURES.inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception as error:
            INFERENCE_FAILURES.inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Champion inference failed",
            ) from error
        finally:
            INFERENCE_LATENCY.observe(perf_counter() - started)
        PREDICTIONS.labels(prediction=str(result.value)).inc()
        return PredictionResponse(
            prediction=result.value,
            churn=bool(result.value),
            churn_probability=result.probability,
            model_name=result.model_name,
            model_version=result.model_version,
            model_alias=result.model_alias,
        )

    @application.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
