import hashlib
import json
from src.types import PipelineRequest


def get_cache_key(req: PipelineRequest) -> str:
    payload = {
        "q": req.query.strip().lower(),
        "m": req.mode,
        "r": req.retrieval,
        "y": req.year,
        "max": req.max_articles_for_facts,
    }
    hash_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()
