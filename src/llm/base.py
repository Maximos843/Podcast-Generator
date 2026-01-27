class LLM:
    def generate(self, prompt: str, system: str = "Ты — полезный ассистент.") -> str:
        raise NotImplementedError