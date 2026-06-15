---
title: Agent Free Food
emoji: 🍕
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
short_description: find free food anywhere around the world
tags:
  - huggingface
  - inference-providers
  - gradio
  - agent
  - exa
  - web-search
  - food
  - events
  - qwen
  - codex
  - track:backyard
  - sponsor:openai
  - achievement:offbrand
  - achievement:fieldnotes
---

# Agent Free Food

🔗 **Live demo:** https://huggingface.co/spaces/build-small-hackathon/Agentfreefood
📦 **Source:** https://github.com/fozagtx/Agentfood
🎥 **Demo video:** https://youtu.be/MMg47HF4oVA
📣 **Social post:** https://www.linkedin.com/posts/fawuzanibrahim_i-built-an-agent-that-finds-free-food-near-ugcPost-7472408995178414080-cNKX/
📝 **Field Notes blog:** https://dev.to/ibrahimpima/i-built-an-agent-that-finds-free-food-near-you-3npb

An agentic chat that finds events with **free food and free drinks** in any city. Type a question — the agent decides whether to chat or to run a live web search, scores each event with an LLM, and returns the best ones.

The agent instructions were prompt-tuned with **Codex** to surface the strongest free-food opportunities instead of generic event listings.

Powered by:
- **Hugging Face Inference Providers** running **Qwen/Qwen2.5-7B-Instruct** through the OpenAI-compatible router
- **Exa** neural web search
- **Gradio** UI

## Build Small Submission

- **Track:** Backyard AI (`track:backyard`)
- **Tracks:** Build Small includes Backyard AI for practical apps and Thousand Token Wood for whimsical apps; Agent Free Food is submitted to the practical Backyard AI track.
- **Sponsor prize:** Best Use of Codex (`sponsor:openai`)
- **Achievements:** Custom UI (`achievement:offbrand`), Field Notes (`achievement:fieldnotes`)
- **Model:** `Qwen/Qwen2.5-7B-Instruct:fastest` via Hugging Face Inference Providers, under the 32B parameter limit
- **Demo video:** https://youtu.be/MMg47HF4oVA
- **Social post:** https://www.linkedin.com/posts/fawuzanibrahim_i-built-an-agent-that-finds-free-food-near-ugcPost-7472408995178414080-cNKX/
- **Field Notes blog:** https://dev.to/ibrahimpima/i-built-an-agent-that-finds-free-food-near-you-3npb
- **Team HF username:** `pima5`

## How it works

1. User asks something in chat ("free food in Austin tonight").
2. The LLM decides whether to call its single tool: `search_free_food(city, threshold)`.
3. If yes, the app runs Exa search across multiple curated queries, fetches page contents, and asks the LLM to score each event 0-100 on free-food likelihood.
4. Events at or above the threshold are streamed back into the chat as a markdown table sorted by score.
5. If no tool is needed, the LLM just chats normally.

## Setup

Set these as Space secrets (Settings → Variables and secrets):

| Name | Required | Value |
|------|----------|-------|
| `HF_TOKEN` | yes | Hugging Face token with **Make calls to Inference Providers** permission |
| `HF_MODEL` | optional | Defaults to `Qwen/Qwen2.5-7B-Instruct:fastest` |
| `HF_ROUTER_BASE_URL` | optional | Defaults to `https://router.huggingface.co/v1` |
| `EXA_API_KEY` | yes | Get one at https://exa.ai |

`MODEL_NAME` is still accepted as a fallback for older deployments, but new setups should use `HF_MODEL`.

## Run locally

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN="hf_..."
export HF_MODEL="Qwen/Qwen2.5-7B-Instruct:fastest"
export EXA_API_KEY="exa-..."

python app.py
```

Open http://127.0.0.1:7860.

## Files

- `app.py` — Gradio UI, tool-routing chat, custom design system
- `free_food_agent.py` — Exa search + LLM scoring pipeline
- `requirements.txt` — `gradio==5.29.0`, `openai>1.0.0`, `exa-py`

## Try it

- "Find me free food events in San Francisco this week"
- "What's happening in NYC tonight with free drinks?"
- "Any free food events in Austin?"
