# yandex_llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from openai import OpenAI


@dataclass
class YandexLLMConfig:
    api_key: str
    folder_id: str
    model_uri: Optional[str] = None
    base_url: str = "https://llm.api.cloud.yandex.net/v1"
    temperature: float = 0.4
    max_tokens: int = 2000
    data_logging_enabled: bool = False
    request_timeout_sec: float = 120.0
    max_retries: int = 3
    retry_sleep_sec: float = 1.5


class YandexGPT5Client:
    def __init__(self, cfg: YandexLLMConfig):
        self.cfg = cfg
        self.model_uri = cfg.model_uri or f"gpt://{cfg.folder_id}/yandexgpt-lite"

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
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_sleep_sec * attempt)
                else:
                    raise last_err
