import time
import uuid
import logging
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.service.metrics import HTTP_LATENCY, HTTP_ERRORS

logger = logging.getLogger("rag-service")


class RequestIdLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = rid

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as e:
            dt = time.perf_counter() - t0
            HTTP_LATENCY.labels(endpoint=request.url.path).observe(dt)
            HTTP_ERRORS.labels(endpoint=request.url.path, type=type(e).__name__).inc()

            logger.exception(
                "request_failed",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(e).__name__,
                },
            )
            raise

        dt = time.perf_counter() - t0
        HTTP_LATENCY.labels(endpoint=request.url.path).observe(dt)

        response.headers["X-Request-Id"] = rid
        logger.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": int(dt * 1000),
            },
        )
        return response
