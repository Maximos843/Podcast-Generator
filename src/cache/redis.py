import json
import logging
from typing import Optional, Any
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, url: str, prefix: str = "podcast:"):
        self.url = url
        self.prefix = prefix
        self._client: Optional[Redis] = None

    async def connect(self):
        try:
            self._client = Redis.from_url(self.url, decode_responses=True)
            await self._client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self._client = None

    async def close(self):
        if self._client:
            await self._client.close()

    async def get(self, key: str) -> Optional[dict]:
        if not self._client: return None
        try:
            data = await self._client.get(f"{self.prefix}{key}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Cache GET error: {e}")
            return None

    async def set(self, key: str, value: dict, ttl: int):
        if not self._client: return
        try:
            await self._client.setex(f"{self.prefix}{key}", ttl, json.dumps(value, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")
