"""
app.routes.health
─────────────────
Mounted at  GET /internal/v1/health
            GET /internal/v1/models

These two endpoints are polled by:
  - The Go service's waitForInference() startup gate
  - The Go /readyz handler (proxied to the client)
  - k3s liveness / readiness probes
  - Prometheus blackbox exporter
"""
from __future__ import annotations

import logging
from typing import Annotated

import torch
from fastapi import APIRouter, Depends, Request

from api.dependency import get_device_info
from core.config import settings
from model.schemas import HealthResponse, ModelsResponse

log = logging.getLogger(__name__)

router = APIRouter()
# No prefix — /internal/v1 is applied by main.py's _register_routers.


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 as long as the process is alive. "
        "Does NOT check whether models have finished loading — use /models for that."
    ),
)
async def health_check() -> HealthResponse:
    return HealthResponse(ok=True)


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="Model load state",
    description=(
        "Reports which model singletons are initialised in app.state. "
        "The Go service logs this at startup and uses it for readiness gating."
    ),
)
async def models_status(
    request: Request,
    device_info: Annotated[dict, Depends(get_device_info)],
) -> ModelsResponse:
    """
    Check each app.state slot set by main._load_models().
    Returns False for any model that failed to initialise or hasn't loaded yet.
    """
    s = request.app.state

    loaded = ModelsResponse(
        ok=True,
        mtcnn            = getattr(s, "mtcnn",          None) is not None,
        vggface2         = getattr(s, "verif_model",    None) is not None,
        dsnt             = getattr(s, "id_detector",    None) is not None,
        gpu              = device_info["gpu"],
        cuda             = device_info["cuda"],
        model_version    = settings.model_version,
        liveness_version = settings.liveness_version,
    )

    log.debug("models probe", extra=loaded.model_dump())
    return loaded
