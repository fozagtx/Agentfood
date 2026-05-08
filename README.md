---
title: AMD HuggingFace Workshop Demo
emoji: 🚀
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
tags:
  - amd
  - amd-hackathon-2026
  - vllm
  - gradio
---

# AMD MI300X AI Demo

A Gradio chat interface connected to a vLLM endpoint running on AMD MI300X GPU.

## Setup

Add these as Space secrets (Settings → Variables and secrets):

| Secret | Value |
|--------|-------|
| `VLLM_BASE_URL` | Your AMD vLLM endpoint, e.g. `http://your-ip:8000/v1` |
| `MODEL_NAME` | Model ID loaded by vLLM, e.g. `Qwen/Qwen2.5-1.5B-Instruct` |
