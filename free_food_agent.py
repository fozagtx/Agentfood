import os
import json
import re
from typing import Any, Iterator

from exa_py import Exa
from openai import OpenAI


DETAIL_PROMPT = """Analyze this event page and estimate the likelihood (0-100) that attendees get free food or drinks.

90-100 -> explicitly offers free food/drinks to attendees
70-89  -> tech/startup/product launch (almost always have food)
50-69  -> networking, mixer, happy hour
30-49  -> community/workshop events
0-29   -> performances, job fairs, lectures (no food expected)

Extract: name, timeAndLocation (date + time + venue), likelihood (int), reasoning (1 sentence), foodAndDrinks (what's offered or empty string)."""


def _build_queries(city: str) -> list[str]:
    c = city.strip() or "San Francisco"
    return [
        f"free food event {c} this week",
        f"tech meetup pizza drinks {c}",
        f"startup demo night free food {c}",
        f"hackathon {c} free meals",
        f"happy hour networking event {c}",
        f"product launch party {c} food drinks",
    ]


def _llm_client() -> OpenAI:
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    return OpenAI(base_url=base_url, api_key=os.environ.get("VLLM_API_KEY", "not-required"))


def _model_name() -> str:
    return os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _score_event(client: OpenAI, model: str, title: str, url: str, content: str) -> dict[str, Any] | None:
    snippet = (content or "")[:6000]
    user_msg = (
        f"{DETAIL_PROMPT}\n\n"
        f"Return ONLY a JSON object with keys: name, timeAndLocation, likelihood, reasoning, foodAndDrinks.\n\n"
        f"URL: {url}\nTITLE: {title}\n\nPAGE CONTENT:\n{snippet}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output strictly valid JSON. No prose."},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    parsed = _extract_json(resp.choices[0].message.content or "")
    if not parsed:
        return None
    try:
        likelihood = int(parsed.get("likelihood", 0))
    except (TypeError, ValueError):
        likelihood = 0
    return {
        "name": parsed.get("name") or title,
        "timeAndLocation": parsed.get("timeAndLocation", ""),
        "likelihood": max(0, min(100, likelihood)),
        "reasoning": parsed.get("reasoning", ""),
        "foodAndDrinks": parsed.get("foodAndDrinks", ""),
        "url": url,
    }


def stream_free_food_events(
    city: str = "San Francisco",
    threshold: int = 70,
    max_results_per_query: int = 8,
    queries: list[str] | None = None,
    exa_api_key: str | None = None,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield (status_message, scored_events_so_far) as work progresses."""
    api_key = exa_api_key or os.environ.get("EXA_API_KEY")
    if not api_key:
        yield "EXA_API_KEY not set.", []
        return

    exa = Exa(api_key=api_key)
    client = _llm_client()
    model = _model_name()

    queries = queries or _build_queries(city)
    seen_urls: set[str] = set()
    scored: list[dict[str, Any]] = []
    total_results = 0

    for i, query in enumerate(queries, 1):
        yield f"[{i}/{len(queries)}] Searching Exa: '{query}'...", sorted(scored, key=lambda e: e["likelihood"], reverse=True)
        try:
            search = exa.search_and_contents(
                query,
                num_results=max_results_per_query,
                type="fast",
                text=True,
            )
        except Exception as e:
            yield f"[{i}/{len(queries)}] Search failed: {e}", sorted(scored, key=lambda e: e["likelihood"], reverse=True)
            continue

        results = list(getattr(search, "results", []) or [])
        total_results += len(results)
        for j, r in enumerate(results, 1):
            url = getattr(r, "url", "") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = getattr(r, "title", "") or ""
            text = getattr(r, "text", "") or ""
            yield (
                f"[{i}/{len(queries)}] Scoring {j}/{len(results)}: {title[:60]}...",
                sorted(scored, key=lambda e: e["likelihood"], reverse=True),
            )
            try:
                event = _score_event(client, model, title, url, text)
            except Exception as e:
                print(f"[score] failed for {url}: {e}")
                continue
            if event and event["likelihood"] >= threshold:
                scored.append(event)

    final = sorted(scored, key=lambda e: e["likelihood"], reverse=True)
    yield (
        f"Done. Searched {len(queries)} queries, scanned {total_results} pages, found {len(final)} events at >= {threshold}%.",
        final,
    )


def find_free_food_events(
    city: str = "San Francisco",
    threshold: int = 70,
    max_results_per_query: int = 8,
    queries: list[str] | None = None,
    exa_api_key: str | None = None,
) -> list[dict[str, Any]]:
    last: list[dict[str, Any]] = []
    for _, events in stream_free_food_events(
        city=city,
        threshold=threshold,
        max_results_per_query=max_results_per_query,
        queries=queries,
        exa_api_key=exa_api_key,
    ):
        last = events
    return last


def events_to_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            e.get("likelihood", 0),
            e.get("name", ""),
            e.get("timeAndLocation", ""),
            e.get("foodAndDrinks", ""),
            e.get("reasoning", ""),
            e.get("url", ""),
        ]
        for e in events
    ]
