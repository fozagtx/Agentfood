import os
import json
import re
import gradio as gr
from openai import OpenAI

from free_food_agent import stream_free_food_events

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-required")


SYSTEM_PROMPT = """You are Agent Food, an assistant running on AMD MI300X GPU via vLLM.

You have ONE tool: search_free_food(city, threshold).
Use it when the user asks about events, free food, free drinks, what's happening, things to do tonight/this week, parties, mixers, hackathons, meetups, or anything event-related in a city.

When you decide to use the tool, respond with ONLY a JSON object on the first line, nothing else:
{"tool": "search_free_food", "city": "<city>", "threshold": <0-100>}

If the user doesn't specify a city, default to San Francisco. If no threshold, use 70.
For any other question, just answer normally as a helpful assistant. Never mention the tool by name to the user."""


def _detect_tool_call(text: str):
    if not text:
        return None
    m = re.search(r'\{[^{}]*"tool"\s*:\s*"search_free_food"[^{}]*\}', text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _events_to_markdown(events, threshold):
    if not events:
        return f"_No events found at ≥ {threshold}% likelihood. Try lowering the threshold or a different city._"
    lines = [
        f"### Found **{len(events)}** event(s) at ≥ {threshold}%",
        "",
        "| Score | Event | When / Where | Food & Drinks | Why |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for e in events:
        name = (e.get("name") or "").replace("|", "\\|")
        when = (e.get("timeAndLocation") or "").replace("|", "\\|")
        food = (e.get("foodAndDrinks") or "").replace("|", "\\|")
        reasoning = (e.get("reasoning") or "").replace("|", "\\|")
        url = e.get("url") or ""
        link = f"[{name}]({url})" if url else name
        lines.append(f"| {e.get('likelihood', 0)} | {link} | {when} | {food} | {reasoning} |")
    return "\n".join(lines)


def _llm_first_pass(history_messages, user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history_messages, {"role": "user", "content": user_message}]
    resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.2)
    return resp.choices[0].message.content or ""


def _llm_stream(history_messages, user_message, system_override=None):
    sys_prompt = system_override or SYSTEM_PROMPT
    messages = [{"role": "system", "content": sys_prompt}, *history_messages, {"role": "user", "content": user_message}]
    stream = client.chat.completions.create(model=MODEL_NAME, messages=messages, stream=True)
    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            partial += delta
            yield partial


def _coerce_history(history):
    out = []
    for item in history or []:
        if isinstance(item, dict):
            out.append({"role": item["role"], "content": item["content"]})
        else:
            out.append({"role": "user", "content": item[0]})
            if item[1]:
                out.append({"role": "assistant", "content": item[1]})
    return out


def chat(message, history):
    history_messages = _coerce_history(history)

    decision = _llm_first_pass(history_messages, message)
    tool_call = _detect_tool_call(decision)

    if tool_call:
        city = (tool_call.get("city") or "San Francisco").strip()
        try:
            threshold = int(tool_call.get("threshold", 70))
        except (TypeError, ValueError):
            threshold = 70
        threshold = max(0, min(100, threshold))

        partial = f"_Searching for free food events in **{city}** (≥ {threshold}%)..._\n\n"
        yield partial

        latest_events = []
        for status, events in stream_free_food_events(city=city, threshold=threshold, max_results_per_query=6):
            latest_events = events
            running_md = _events_to_markdown(events, threshold) if events else ""
            yield f"_{status}_\n\n{running_md}".rstrip()

        yield _events_to_markdown(latest_events, threshold)
        return

    yield from _llm_stream(history_messages, message)


DESIGN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Instrument+Serif&display=swap');

:root {
    --navy: rgb(0, 34, 89);
    --brand-blue: rgb(38, 112, 220);
    --steel-muted: rgb(121, 138, 166);
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
.gr-button.primary, button.primary, .gr-button.primary * { color: #ffffff !important; }

h1, h2, h3, h4, .prose h1, .prose h2, .prose h3 {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-weight: 400 !important;
    color: var(--navy) !important;
    letter-spacing: 0 !important;
}
h1 { font-size: 32px !important; }
h2, h3 { font-size: 24px !important; }

label, .gr-box label {
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
    border-radius: 16px !important;
    height: 46px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
.gr-button.primary, button.primary {
    background: var(--action-gradient) !important;
    color: #fff !important;
}
.gr-button.secondary, button.secondary {
    background: #fff !important;
    color: var(--brand-blue) !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
    box-shadow: var(--inset-glow) -2px -2px 4px 0px inset, var(--inset-glow) 2px 2px 4px 0px inset !important;
}

input[type="text"], textarea, .gr-textbox textarea, .gr-textbox input {
    background: #fff !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
    border-radius: 12px !important;
    color: var(--navy) !important;
    box-shadow: var(--inset-glow) -2px -2px 4px 0px inset, var(--inset-glow) 2px 2px 4px 0px inset !important;
}

.chatbot, [data-testid="chatbot"], .bubble-wrap, .message-wrap {
    background: var(--card-wash) !important;
    border-radius: 12px !important;
    box-shadow: rgba(255,255,255,0.75) -4px -4px 6px 0px inset, rgba(255,255,255,0.75) 4px 4px 6px 0px inset !important;
}
.message, .message-bubble, .chatbot .message, .chatbot .bot, .chatbot [data-testid="bot"] {
    background: #ffffff !important;
    color: var(--navy) !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
    box-shadow: var(--inset-glow) -2px -2px 4px 0px inset, var(--inset-glow) 2px 2px 4px 0px inset !important;
    border-radius: 12px !important;
}
.chatbot .user, .chatbot [data-testid="user"] {
    background: var(--chip-active) !important;
    color: var(--brand-blue) !important;
    border-radius: 12px !important;
}

table { background: #fff !important; border-radius: 12px !important; overflow: hidden; }
th { background: var(--card-wash) !important; font-weight: 600 !important; }
td { font-size: 14px !important; }
"""


with gr.Blocks(title="Agent Food", theme=gr.themes.Soft(), css=DESIGN_CSS) as demo:
    gr.Markdown("# Agent Food")
    gr.ChatInterface(
        fn=chat,
        type="messages",
        examples=[
            "Find me free food events in San Francisco this week",
            "What's happening in NYC tonight with free drinks?",
            "Any free food events in Austin?",
            "Explain what AMD MI300X is.",
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
