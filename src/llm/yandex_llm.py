# yandex_llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from openai import OpenAI
import logging

logger = logging.getLogger("rag-llm")


@dataclass
class YandexLLMConfig:
    api_key: str
    folder_id: str
    model_uri: Optional[str] = None
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

        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            project=cfg.folder_id,
            default_headers=default_headers or None,
            timeout=cfg.request_timeout_sec,
        )

    def generate(self, prompt: str, system: str = "Ты — полезный ассистент.") -> str:
        last_err = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_uri,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = resp.choices[0].message.content

                usage = getattr(resp, "usage", None)
                if usage is not None:
                    # openai SDK обычно даёт usage.prompt_tokens / completion_tokens / total_tokens
                    logger.info(
                        "llm_usage",
                        extra={
                            "model": self.model_uri,
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                            "max_tokens": self.cfg.max_tokens,
                            # стадия берётся из system (мы добавили STAGE:...):
                            "stage": system.splitlines()[0].replace("STAGE:", "").strip() if system.startswith("STAGE:") else "unknown",
                            "prompt_chars": len(prompt),
                            "system_chars": len(system or ""),
                        },
                    )

                return content
            except Exception as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_sleep_sec * attempt)
                else:
                    raise last_err
