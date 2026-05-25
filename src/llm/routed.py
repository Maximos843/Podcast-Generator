from dataclasses import dataclass

from src.llm.base import LLM


@dataclass(frozen=True)
class LLMTaskRoutingConfig:
    facts_temperature: float = 0.1
    factcheck_temperature: float = 0.0
    strict_refs_temperature: float = 0.0
    script_temperature: float = 0.5
    repair_temperature: float = 0.2


class RoutedLLM(LLM):
    def __init__(
        self,
        *,
        weak_llm: LLM,
        strong_llm: LLM | None = None,
        routing: LLMTaskRoutingConfig | None = None,
    ):
        self.weak_llm = weak_llm
        self.strong_llm = strong_llm or weak_llm
        self.routing = routing or LLMTaskRoutingConfig()

    def _pick_llm(self, task: str | None) -> LLM:
        if task in {"script", "repair"}:
            return self.strong_llm
        return self.weak_llm

    def _pick_temperature(self, task: str | None, temperature: float | None) -> float:
        if temperature is not None:
            return temperature
        if task == "facts":
            return self.routing.facts_temperature
        if task == "factcheck":
            return self.routing.factcheck_temperature
        if task == "strict_refs":
            return self.routing.strict_refs_temperature
        if task == "script":
            return self.routing.script_temperature
        if task == "repair":
            return self.routing.repair_temperature
        return 0.1

    async def generate(
        self,
        prompt: str,
        system: str = "Ты — полезный ассистент.",
        *,
        task: str | None = None,
        temperature: float | None = None,
    ) -> str:
        llm = self._pick_llm(task)
        used_temperature = self._pick_temperature(task, temperature)
        return await llm.generate(
            prompt,
            system=system,
            task=task,
            temperature=used_temperature,
        )
