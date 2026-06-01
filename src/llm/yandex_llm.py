from dataclasses import dataclass
import time
import logging
import asyncio

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger("rag-llm")


@dataclass
class YandexLLMConfig:
    api_key: str
    folder_id: str
    model_uri: str | None = None
    base_url: str = "https://llm.api.cloud.yandex.net/v1"
    temperature: float = 0.1
    max_tokens: int = 4096
    data_logging_enabled: bool = False
    request_timeout_sec: float = 120.0
    max_retries: int = 3
    retry_sleep_sec: float = 1.5


class YandexGPT5Client:
    def __init__(self, cfg: YandexLLMConfig):
        self.cfg = cfg
        self.model_uri = cfg.model_uri or f"gpt://{cfg.folder_id}/yandexgpt-5-lite"

        default_headers = {}
        if cfg.data_logging_enabled is False:
            default_headers["x-data-logging-enabled"] = "false"

        self.client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            project=cfg.folder_id,
            default_headers=default_headers or None,
            timeout=cfg.request_timeout_sec,
        )

    async def generate(
        self,
        prompt: str,
        system: str = "Ты — полезный ассистент.",
        *,
        task: str | None = None,
        temperature: float | None = None,
    ) -> str:  # type: ignore
        last_err = None
        used_temperature = self.cfg.temperature if temperature is None else temperature

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_uri,
                    temperature=used_temperature,
                    max_tokens=self.cfg.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )

                content = resp.choices[0].message.content
                usage = getattr(resp, "usage", None)

                if usage is not None:
                    logger.info(
                        "llm_usage",
                        extra={
                            "model": self.model_uri,
                            "task": task or "unknown",
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                            "max_tokens": self.cfg.max_tokens,
                            "temperature": used_temperature,
                            "prompt_chars": len(prompt),
                            "system_chars": len(system or ""),
                        },
                    )
                return content  # type: ignore

            except Exception as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_sleep_sec * attempt)
                else:
                    raise last_err
