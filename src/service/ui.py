import gradio as gr
from src.config import AppConfig
from src.pipeline.service import PipelineService
from src.types import PipelineRequest


def create_ui(cfg: AppConfig, pipeline: PipelineService):
    """Создаёт простой Gradio интерфейс."""

    def generate(query: str, year: int | None, mode: str, retrieval: str, max_articles: int, include_debug: bool):
        req = PipelineRequest(
            query=query,
            year=year if year else None,
            mode=mode,  # type: ignore
            retrieval=retrieval,  # type: ignore
            max_articles_for_facts=max_articles,
            include_debug=include_debug,
        )
        result = pipeline.generate(req)

        facts = "\n".join([f"- {f.statement}" for card in result.fact_cards for f in card.facts])
        warnings = ""  #"\n".join([f"⚠️ {u.claim}" for u in (result.fact_check.unsupported or [])])
        debug = str(result.debug) if include_debug and result.debug else ""

        return result.script, facts, warnings, debug

    with gr.Blocks(title="🎙️ Podcast Generator") as demo:
        gr.Markdown("## 🎙️ Генератор исторических подкастов")

        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="Тема", lines=3, placeholder="Например: реформы Петра I")
                year = gr.Number(label="Год (опционально)", precision=0)
                mode = gr.Radio(["quality", "fast"], value="quality", label="Режим")
                retrieval = gr.Radio(["hybrid", "dense"], value="hybrid", label="Поиск")
                max_articles = gr.Slider(3, 15, value=7, step=1, label="Источников")
                include_debug = gr.Checkbox(label="Отладка")
                btn = gr.Button("🚀 Сгенерировать", variant="primary")

            with gr.Column():
                script_out = gr.Markdown(label="📝 Сценарий")
                facts_out = gr.Markdown(label="🎯 Факты")
                warnings_out = gr.Markdown(label="⚠️ Проблемы")
                debug_out = gr.Code(label="🔧 Отладка", language="json")

        btn.click(
            fn=generate,
            inputs=[query, year, mode, retrieval, max_articles, include_debug],
            outputs=[script_out, facts_out, warnings_out, debug_out]
        )

    return demo