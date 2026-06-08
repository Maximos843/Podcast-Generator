# load_tests/locustfile_fast_only.py

import random
import uuid

from locust import HttpUser, task, between


QUERIES = [
    "Индустриализация 1930-х годов",
    "Жизнь рабочих в советских газетах",
    "Сельское хозяйство в СССР",
    "Культурная жизнь 1930-х",
    "Политическая повестка советской прессы",
]


class FastPipelineUser(HttpUser):
    wait_time = between(3, 5)

    @task
    def generate_fast(self):
        payload = {
            "query": random.choice(QUERIES),
            "mode": "fast",
            "retrieval": "hybrid",
            "max_articles_for_facts": 5,
            "include_debug": False,
        }

        headers = {
            "X-Request-Id": f"locust-fast-{uuid.uuid4()}",
        }

        with self.client.post(
            "/generate",
            json=payload,
            headers=headers,
            name="/generate fast hybrid",
            catch_response=True,
            timeout=240,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:1000]}")
                return

            try:
                data = resp.json()
            except Exception as e:
                resp.failure(f"Invalid JSON: {e}; body={resp.text[:1000]}")
                return

            if not data.get("script"):
                resp.failure("Empty script")
                return

            if not data.get("hits"):
                resp.failure("Empty hits")
                return

            if not data.get("fact_cards"):
                resp.failure("Empty fact_cards")
                return

            timings = data.get("timings") or {}
            if not timings.get("total_ms"):
                resp.failure(f"Bad timings: {timings}")
                return

            resp.success()