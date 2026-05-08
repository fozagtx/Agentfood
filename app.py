import os
import gradio as gr
from openai import OpenAI

from free_food_agent import stream_free_food_events, events_to_rows

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-required")


def chat(message, history):
    messages = [{"role": "system", "content": "You are Qwen, a helpful assistant created by Alibaba Cloud, running on AMD MI300X GPU via vLLM. You are not Claude, ChatGPT, or any other model."}]
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            messages.append({"role": "user", "content": item[0]})
            if item[1]:
                messages.append({"role": "assistant", "content": item[1]})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            partial += delta
            yield partial


DESIGN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Instrument+Serif&display=swap');

:root {
    --navy: rgb(0, 34, 89);
    --brand-blue: rgb(38, 112, 220);
    --steel-muted: rgb(121, 138, 166);
    --stats-grey: rgb(143, 159, 184);
    --card-wash: rgb(239, 244, 249);
    --chip-active: rgb(215, 231, 254);
    --chip-active-border: rgb(235, 243, 254);
    --chip-inactive: rgb(242, 243, 246);
    --inset-glow: rgb(235, 243, 255);
    --action-gradient: linear-gradient(rgb(0, 68, 185) 5.5%, rgb(0, 116, 236) 35%, rgb(78, 177, 255) 65%, rgb(173, 217, 255) 95%);
}

.gradio-container, body, .gradio-container * {
    font-family: 'Instrument Sans', system-ui, sans-serif !important;
    color: var(--navy) !important;
}
.gradio-container, body {
    background: linear-gradient(rgb(189, 215, 255) 0%, rgb(255, 255, 255) 39.45%) fixed !important;
}
.gradio-container p, .gradio-container span, .gradio-container li,
.gradio-container td, .gradio-container th, .gradio-container .prose,
.gradio-container .prose *, .markdown, .markdown * {
    color: var(--navy) !important;
}
.gr-button.primary, button.primary, .gr-button.primary * {
    color: #ffffff !important;
}

h1, h2, h3, h4, .prose h1, .prose h2, .prose h3 {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-weight: 400 !important;
    color: var(--navy) !important;
    letter-spacing: 0 !important;
}

h1 { font-size: 32px !important; }
h2, h3 { font-size: 24px !important; }

label, .gr-box label, span[data-testid="block-info"] {
    font-family: 'Instrument Sans', sans-serif !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
    letter-spacing: -0.5px !important;
}

.gr-button, button.lg, button.sm, button.primary, button.secondary {
    font-family: 'Instrument Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: -0.5px !important;
    line-height: 18px !important;
    border-radius: 16px !important;
    height: 46px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    overflow: hidden !important;
    transition: transform 0.12s ease;
}
.gr-button:hover { transform: translateY(-1px); }

.gr-button.primary, button.primary {
    background: var(--action-gradient) !important;
    color: #fff !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.3) inset !important;
}

.gr-button.secondary, button.secondary {
    background: #fff !important;
    color: var(--brand-blue) !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
    box-shadow:
        var(--inset-glow) -2px -2px 4px 0px inset,
        var(--inset-glow) 2px 2px 4px 0px inset !important;
}

.gr-form, .gr-box, .block, .form, fieldset {
    background: var(--card-wash) !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow:
        rgba(255,255,255,0.75) -4px -4px 6px 0px inset,
        rgba(255,255,255,0.75) 4px 4px 6px 0px inset !important;
    padding: 12px !important;
}

input[type="text"], input[type="password"], textarea, .gr-textbox textarea, .gr-textbox input {
    background: #fff !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
    border-radius: 12px !important;
    font-family: 'Instrument Sans', sans-serif !important;
    color: var(--navy) !important;
    box-shadow:
        var(--inset-glow) -2px -2px 4px 0px inset,
        var(--inset-glow) 2px 2px 4px 0px inset !important;
}

button[role="tab"] {
    border-radius: 9999px !important;
    height: 34px !important;
    padding: 8px 14px !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    background: var(--chip-inactive) !important;
    color: var(--steel-muted) !important;
    border: 1px solid rgb(247,247,249) !important;
    margin-right: 6px !important;
}
button[role="tab"][aria-selected="true"] {
    background: var(--chip-active) !important;
    color: var(--brand-blue) !important;
    border: 1px solid var(--chip-active-border) !important;
    box-shadow:
        rgba(255,255,255,0.75) -2px -2px 4px inset,
        rgba(255,255,255,0.75) 2px 2px 4px inset !important;
}

.chatbot, .chatbot > div, [data-testid="chatbot"], .bubble-wrap, .message-wrap {
    background: var(--card-wash) !important;
    border-radius: 12px !important;
    box-shadow:
        rgba(255,255,255,0.75) -4px -4px 6px 0px inset,
        rgba(255,255,255,0.75) 4px 4px 6px 0px inset !important;
    color: var(--navy) !important;
}

.message, .message-bubble, .chatbot .message, .chatbot .user, .chatbot .bot,
.chatbot [data-testid="user"], .chatbot [data-testid="bot"] {
    border-radius: 12px !important;
    background: #ffffff !important;
    color: var(--navy) !important;
    box-shadow:
        var(--inset-glow) -2px -2px 4px 0px inset,
        var(--inset-glow) 2px 2px 4px 0px inset !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
}

.chatbot .user, .chatbot [data-testid="user"] {
    background: var(--chip-active) !important;
    color: var(--brand-blue) !important;
}

.message *, .message-bubble *, .chatbot .message *,
.chatbot .user *, .chatbot .bot *, .chatbot p, .chatbot span, .chatbot div {
    color: inherit !important;
    background: transparent !important;
}

.gradio-container input[type="range"] {
    accent-color: var(--brand-blue);
}

table { background: #fff !important; border-radius: 12px !important; overflow: hidden; }
th { background: var(--card-wash) !important; color: var(--navy) !important; font-family: 'Instrument Sans', sans-serif !important; font-weight: 600 !important; }
td { color: var(--navy) !important; font-size: 14px !important; }
"""


EVENT_COLUMNS = ["Score", "Event", "When / Where", "Food & Drinks", "Reasoning", "URL"]


def run_free_food_agent(city, threshold, results_per_query):
    threshold = int(threshold)
    results_per_query = int(results_per_query)
    city = (city or "San Francisco").strip()

    for status, events in stream_free_food_events(
        city=city,
        threshold=threshold,
        max_results_per_query=results_per_query,
    ):
        rows = events_to_rows(events)
        count_md = f"### Found **{len(events)}** event(s) at >= {threshold}%"
        yield status, count_md, rows


with gr.Blocks(title="Agent Food", theme=gr.themes.Soft(), css=DESIGN_CSS) as demo:
    gr.Markdown("# Agent Food")

    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=chat,
            type="messages",
            description="Chat with an LLM running on AMD MI300X GPU via vLLM.",
            examples=["Explain what AMD MI300X is.", "Write a Python hello world."],
            cache_examples=False,
        )

    with gr.Tab("Free Food Agent"):
        gr.Markdown(
            "Discovers local events via **Exa search** (neural web search across the open web) "
            "and scores each one with the local vLLM model using a free-food likelihood rubric."
        )

        with gr.Row():
            with gr.Column(scale=2):
                city = gr.Textbox(value="San Francisco", label="City")
            with gr.Column(scale=1):
                threshold = gr.Slider(0, 100, value=70, step=5, label="Min likelihood %")
            with gr.Column(scale=1):
                per_query = gr.Slider(1, 15, value=8, step=1, label="Results per query")

        with gr.Row():
            run_btn = gr.Button("Find free food events", variant="primary", scale=3)
            clear_btn = gr.Button("Clear", scale=1)

        status = gr.Textbox(label="Status", interactive=False, lines=1)
        count_md = gr.Markdown()
        results = gr.Dataframe(
            headers=EVENT_COLUMNS,
            datatype=["number", "str", "str", "str", "str", "str"],
            wrap=True,
            interactive=False,
            label="Events (sorted by score)",
        )

        run_btn.click(
            run_free_food_agent,
            [city, threshold, per_query],
            [status, count_md, results],
        )
        clear_btn.click(lambda: ("", "", []), None, [status, count_md, results])


if __name__ == "__main__":
    demo.launch()
