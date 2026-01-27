# src/service/app.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import AppConfig
from src.pipeline.service import PipelineService
from src.domain.types import PipelineRequest
from src.service.schemas import GenerateRequest, GenerateResponse
from src.service.wiring import build_deps


def create_app() -> FastAPI:
    cfg = AppConfig.from_env()
    deps = build_deps(cfg)
    service = PipelineService(deps=deps)

    app = FastAPI(title="RAG Podcast Service", version="0.1.0")

    @app.get("/health")
    def health():
        # минимальная проверка доступности Qdrant
        try:
            deps.client.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False
        return {"status": "ok" if qdrant_ok else "degraded", "qdrant": qdrant_ok}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(body: GenerateRequest):
        req = PipelineRequest(
            query=body.query,
            year=body.year,
            mode=body.mode,
            retrieval=body.retrieval,
            max_articles_for_facts=body.max_articles_for_facts,
            include_debug=body.include_debug,
        )
        resp = service.generate(req)
        # FastAPI сам сериализует pydantic-модель response_model, но у нас resp — dataclass.
        # Поэтому отдаём через .__dict__ + ручная сборка pydantic-ответа:
        out = GenerateResponse(
            request_id=resp.request_id,
            hits=[
                {
                    "article_id": h.article_id,
                    "score": h.score,
                    "year": h.year,
                    "best_chunks": [
                        {"chunk_id": c.chunk_id, "text": c.text, "score": c.score, "year": c.year}
                        for c in h.best_chunks
                    ],
                }
                for h in resp.hits
            ],
            fact_cards=resp.fact_cards,
            outline=resp.outline,
            script=resp.script,
            fact_check=resp.fact_check,
            timings=resp.timings.__dict__,
            debug=resp.debug,
        )
        return JSONResponse(content=out.model_dump())

    return app


app = create_app()
