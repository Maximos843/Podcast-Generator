# src/service/app.py
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import AppConfig
from src.types import PipelineRequest, PipelineResponse
from src.pipeline.service import PipelineService
from src.service.wiring import build_deps
from src.service.middleware import RequestIdLoggingMiddleware
from src.service.metrics import GENERATE_LATENCY, GENERATE_ERRORS


def _setup_logging(env: str) -> None:
    level = logging.DEBUG if env == "dev" else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app() -> FastAPI:
    cfg = AppConfig.from_env()
    _setup_logging(cfg.env)

    deps = build_deps(cfg)
    service = PipelineService(deps)

    app = FastAPI(title="RAG Podcast Service", version="0.4.0")
    app.add_middleware(RequestIdLoggingMiddleware)

    @app.get("/health")
    def health():
        qdrant_ok = True
        try:
            deps.client.get_collections()
        except Exception:
            qdrant_ok = False
        return {"status": "ok" if qdrant_ok else "degraded", "qdrant": qdrant_ok}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate", response_model=PipelineResponse)
    def generate(body: PipelineRequest, request: Request) -> PipelineResponse:
        rid = getattr(request.state, "request_id", None)

        t0 = time.perf_counter()
        try:
            resp = service.generate(body, request_id=rid)
        except Exception as e:
            GENERATE_ERRORS.labels(
                mode=body.mode,
                retrieval=body.retrieval,
                type=type(e).__name__,
            ).inc()
            raise
        finally:
            dt = time.perf_counter() - t0
            GENERATE_LATENCY.labels(mode=body.mode, retrieval=body.retrieval).observe(dt)

        return resp

    return app


app = create_app()
