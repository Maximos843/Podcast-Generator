import os
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from gradio import mount_gradio_app

from src.config import AppConfig
from src.types import PipelineRequest, PipelineResponse
from src.pipeline.service import PipelineService
from src.service.wiring import build_deps
from src.service.middleware import RequestIdLoggingMiddleware
from src.service.metrics import GENERATE_LATENCY, GENERATE_ERRORS
from src.service.ui import create_ui

# Импорты для Redis-кэша
from src.cache.redis import RedisCache
from src.cache.utils import get_cache_key


def _setup_logging(env: str) -> None:
    level = logging.DEBUG if env == "dev" else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app() -> FastAPI:
    cfg = AppConfig.from_env()
    _setup_logging(cfg.env)

    # 1. Собираем зависимости ОДИН раз
    deps = build_deps(cfg)

    # 2. Инициализируем Redis-кэш (если включен в конфиге)
    cache = None
    if getattr(cfg, "cache_enabled", False):
        cache = RedisCache(url=getattr(cfg, "redis_url", "redis://localhost:6379"), prefix="podcast:")

    # 3. Передаем кэш в сервис
    service = PipelineService(deps, cache=cache)

    app = FastAPI(title="RAG Podcast Service", version="0.4.0")
    app.add_middleware(RequestIdLoggingMiddleware)
    
    # Сохраняем кэш в state приложения для админ-эндпоинтов
    app.state.cache = cache

    @app.get("/health")
    def health():
        qdrant_ok = True
        try:
            deps.client.get_collections()
        except Exception:
            qdrant_ok = False
        
        cache_ok = app.state.cache is not None and app.state.cache.is_available()
        
        return {
            "status": "ok" if (qdrant_ok and cache_ok) else "degraded", 
            "qdrant": qdrant_ok,
            "cache": cache_ok
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate", response_model=PipelineResponse)
    async def generate(body: PipelineRequest, request: Request) -> PipelineResponse:
        rid = getattr(request.state, "request_id", None)
        t0 = time.perf_counter()
        
        try:
            # Вызываем асинхронный метод сервиса (внутри него уже есть логика проверки кэша)
            resp = await service.generate(body, request_id=rid)
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

    # =========================================================
    # Админ-эндпоинты для проверки кэша (удобно для тестов)
    # =========================================================
    @app.get("/cache/stats")
    async def cache_stats():
        if app.state.cache and app.state.cache.is_available():
            return {"status": "connected", "url": getattr(cfg, "redis_url", "redis://localhost:6379")}
        return {"status": "disabled or disconnected"}

    @app.delete("/cache")
    async def clear_cache(pattern: str = "*"):
        if not app.state.cache or not app.state.cache.is_available():
            return {"error": "Cache is not available"}
        
        # Удаляем все ключи с префиксом podcast:
        count = await app.state.cache.clear_pattern(pattern)
        return {"cleared_keys_count": count, "pattern": pattern}

    # =========================================================
    # События запуска и остановки (для подключения к Redis)
    # =========================================================
    @app.on_event("startup")
    async def startup_event():
        if app.state.cache:
            await app.state.cache.connect()
            logging.info("✅ Redis cache initialization complete.")

    @app.on_event("shutdown")
    async def shutdown_event():
        if app.state.cache:
            await app.state.cache.close()
            logging.info("🔌 Redis cache connection closed.")

    # =========================================================
    # ИСПРАВЛЕНИЕ: Убран двойной build_deps!
    # Используем уже созданный экземпляр `service`
    # =========================================================
    if os.getenv("ENABLE_UI", "false").lower() == "true":
        logging.info("🎨 Mounting Gradio UI at /ui")
        demo = create_ui(cfg, service)
        mount_gradio_app(app, demo, path="/ui")

    return app


app = create_app()