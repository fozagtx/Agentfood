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

    try:
        decision = _llm_first_pass(history_messages, message)
    except Exception as e:
        yield (
            f"⚠️ Couldn't reach the model.\n\n"
            f"`VLLM_BASE_URL = {VLLM_BASE_URL}`\n\n"
            f"Set `VLLM_BASE_URL` and `MODEL_NAME` in HF Space → Settings → Variables and secrets, "
            f"pointing at a publicly reachable vLLM endpoint.\n\n"
            f"_Error: {type(e).__name__}: {e}_"
        )
        return

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

    try:
        yield from _llm_stream(history_messages, message)
    except Exception as e:
        yield f"⚠️ Streaming failed: {type(e).__name__}: {e}"


DESIGN_CSS = """
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
    font-family: 'Instrument Sans', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
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

.examples,
.examples-holder,
[class*="examples"] {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    border: none !important;
}

.examples button,
.examples-holder button,
[class*="examples"] button,
.example,
button.example,
[data-testid="example"] {
    background: var(--chip-active) !important;
    color: var(--brand-blue) !important;
    border: 1px solid var(--chip-active-border) !important;
    border-radius: 9999px !important;
    height: auto !important;
    min-height: 34px !important;
    padding: 8px 14px !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: -0.5px !important;
    box-shadow:
        rgba(255,255,255,0.75) -2px -2px 4px inset,
        rgba(255,255,255,0.75) 2px 2px 4px inset !important;
    transition: transform 0.12s ease;
}
.examples button *,
.examples-holder button *,
[class*="examples"] button *,
.example *,
button.example *,
[data-testid="example"] * {
    color: var(--brand-blue) !important;
    background: transparent !important;
}
.examples button:hover,
[class*="examples"] button:hover { transform: translateY(-1px); }

/* Kill the indigo "Chatbot" label badge */
.chatbot .label-wrap,
.chatbot .label,
[data-testid="chatbot"] .label-wrap,
[data-testid="chatbot"] .label,
.chatbot > div > .svelte-* {
    display: none !important;
}
.chatbot label, [data-testid="chatbot"] label {
    background: transparent !important;
    color: var(--steel-muted) !important;
    padding: 0 !important;
    font-size: 12px !important;
}

/* Message input bar — was rendering dark */
.gradio-container footer { display: none !important; }
[data-testid="textbox"], .input-row, .input-container,
.chat-input, .chat-input-container, footer.svelte-* {
    background: var(--card-wash) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
    padding: 6px !important;
    box-shadow: rgba(255,255,255,0.75) -2px -2px 4px inset, rgba(255,255,255,0.75) 2px 2px 4px inset !important;
}
.input-row textarea, .chat-input textarea,
[data-testid="textbox"] textarea {
    background: #fff !important;
    color: var(--navy) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 37, 97, 0.06) !important;
    box-shadow: var(--inset-glow) -2px -2px 4px inset, var(--inset-glow) 2px 2px 4px inset !important;
}
.input-row textarea::placeholder, .chat-input textarea::placeholder {
    color: var(--steel-muted) !important;
}

/* Send button (paper plane) */
button[aria-label="Submit"], button.send-button, .submit-button, button[title="Submit"] {
    background: var(--action-gradient) !important;
    color: #fff !important;
    border-radius: 12px !important;
    height: 40px !important;
    width: 40px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}
button[aria-label="Submit"] svg, button.send-button svg {
    color: #fff !important;
    stroke: #fff !important;
}

/* Layout: full width, comfortable padding */
.gradio-container { max-width: 960px !important; margin: 0 auto !important; padding: 24px 16px !important; }
.contain, .main, .wrap { max-width: 100% !important; }

@media (max-width: 640px) {
    h1 { font-size: 26px !important; }
    .gradio-container { padding: 16px 12px !important; }
}

/* Nuclear: any wrapper around the chat input that's still dark */
.gradio-container [class*="input"]:not(textarea):not(input),
.gradio-container [class*="Input"]:not(textarea):not(input),
.gradio-container [class*="Container"],
.gradio-container .panel {
    background: var(--card-wash) !important;
}
.gradio-container [style*="background"] { background: inherit !important; }

/* Send/submit button — round it, blue-gradient it */
button.svelte-1ipelgc, button.send-btn, button[aria-label*="Send"], button[aria-label*="Submit"],
.input-row button, [data-testid="textbox"] + button {
    background: var(--action-gradient) !important;
    color: #fff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    width: 44px !important;
    height: 44px !important;
}
button.svelte-1ipelgc *, button.send-btn *, button[aria-label*="Send"] *, button[aria-label*="Submit"] * {
    color: #fff !important;
    fill: #fff !important;
    stroke: #fff !important;
}

/* Hide any stray indigo pill labels */
.label-wrap[style*="indigo"], .label-wrap[style*="rgb(99"] { display: none !important; }
"""


FONTS_HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Instrument+Serif&display=swap" rel="stylesheet">'
)


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Instrument Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(rgb(189,215,255) 0%, rgb(255,255,255) 39.45%)",
    body_background_fill_dark="linear-gradient(rgb(189,215,255) 0%, rgb(255,255,255) 39.45%)",
    body_text_color="rgb(0, 34, 89)",
    body_text_color_dark="rgb(0, 34, 89)",
    background_fill_primary="rgb(239, 244, 249)",
    background_fill_primary_dark="rgb(239, 244, 249)",
    background_fill_secondary="rgb(255, 255, 255)",
    background_fill_secondary_dark="rgb(255, 255, 255)",
    block_background_fill="rgb(239, 244, 249)",
    block_background_fill_dark="rgb(239, 244, 249)",
    block_label_background_fill="rgb(215, 231, 254)",
    block_label_background_fill_dark="rgb(215, 231, 254)",
    block_label_text_color="rgb(38, 112, 220)",
    block_label_text_color_dark="rgb(38, 112, 220)",
    input_background_fill="rgb(255, 255, 255)",
    input_background_fill_dark="rgb(255, 255, 255)",
    input_border_color="rgba(0, 37, 97, 0.06)",
    input_border_color_dark="rgba(0, 37, 97, 0.06)",
    button_primary_background_fill="linear-gradient(rgb(0,68,185) 5.5%, rgb(0,116,236) 35%, rgb(78,177,255) 65%, rgb(173,217,255) 95%)",
    button_primary_background_fill_dark="linear-gradient(rgb(0,68,185) 5.5%, rgb(0,116,236) 35%, rgb(78,177,255) 65%, rgb(173,217,255) 95%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="rgb(255, 255, 255)",
    button_secondary_background_fill_dark="rgb(255, 255, 255)",
    button_secondary_text_color="rgb(38, 112, 220)",
    button_secondary_text_color_dark="rgb(38, 112, 220)",
    border_color_primary="rgba(0, 37, 97, 0.06)",
    border_color_primary_dark="rgba(0, 37, 97, 0.06)",
    color_accent="rgb(38, 112, 220)",
    color_accent_soft="rgb(215, 231, 254)",
    color_accent_soft_dark="rgb(215, 231, 254)",
)


with gr.Blocks(title="Agent Food", theme=THEME, css=DESIGN_CSS, head=FONTS_HEAD) as demo:
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
