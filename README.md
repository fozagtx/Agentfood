---
title: Agent Free Food
emoji: 🍕
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
tags:
  - amd
  - amd-hackathon-2026
  - vllm
  - gradio
  - agent
  - exa
---

# Agent Free Food

🔗 **Live demo:** https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/AgentFreeFood

An agentic chat that finds events with **free food and free drinks** in any city. Type a question — the agent decides whether to chat or to run a live web search, scores each event with an LLM, and returns the best ones.

Powered by:
- **AMD MI300X GPU** running **Qwen2.5-14B-Instruct** via **vLLM** (OpenAI-compatible)
- **Exa** neural web search
- **Gradio** UI

## How it works

1. User asks something in chat ("free food in Austin tonight").
2. The LLM decides whether to call its single tool: `search_free_food(city, threshold)`.
3. If yes, the app runs Exa search across multiple curated queries, fetches page contents, and asks the LLM to score each event 0–100 on free-food likelihood.
4. Events at or above the threshold are streamed back into the chat as a markdown table sorted by score.
5. If no tool is needed, the LLM just chats normally.

## Setup

Set these as Space secrets (Settings → Variables and secrets):

| Name | Required | Value |
|------|----------|-------|
| `VLLM_BASE_URL` | yes | Public vLLM endpoint, e.g. `http://YOUR_IP:8000/v1` |
| `MODEL_NAME` | yes | e.g. `Qwen/Qwen2.5-14B-Instruct` |
| `EXA_API_KEY` | yes | Get one at https://exa.ai |
| `VLLM_API_KEY` | optional | Only if your vLLM was started with `--api-key` |

## Run locally

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export VLLM_BASE_URL="http://localhost:8000/v1"   # via SSH tunnel or local vLLM
export MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
export EXA_API_KEY="exa-..."

python app.py
```

Open http://127.0.0.1:7860.

## Files

- `app.py` — Gradio UI, tool-routing chat, custom design system
- `free_food_agent.py` — Exa search + LLM scoring pipeline
- `requirements.txt` — `gradio>=5.0.0`, `openai>1.0.0`, `exa-py`

## Try it

- "Find me free food events in San Francisco this week"
- "What's happening in NYC tonight with free drinks?"
- "Any free food events in Austin?"
