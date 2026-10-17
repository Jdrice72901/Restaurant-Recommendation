"""Write the final recommendation text from the ranked restaurants.

The LLM only sees the top restaurants and their matching review snippets,
never the whole dataset. The provider is picked in config.yaml:
claude (default, needs ANTHROPIC_API_KEY), ollama (local), or none.
"""

import json
import os
from collections.abc import Iterator


class MissingAPIKeyError(RuntimeError):
    pass

SYSTEM_PROMPT = """You are a restaurant recommendation assistant built on Yelp reviews.

You will get a user's request and a numbered list of candidate restaurants, each with its \
rating, price, categories, and review excerpts that matched the request.

Recommend the best 3 to 5 restaurants for the request, using only restaurants from the list. \
For each one, say in a sentence or two why it fits, grounded in what the review excerpts \
actually say. If none of the candidates fit well, say so plainly instead of stretching. \
Do not invent restaurants, dishes, or facts that are not in the excerpts. Keep it concise."""


def build_prompt(query: str, restaurants: list[dict]) -> str:
    lines = [f"Request: {query}", "", "Candidates:"]
    for i, r in enumerate(restaurants, 1):
        price = "$" * r["price"] if r.get("price") else "unknown"
        lines.append(
            f"{i}. {r['name']} ({r['city']}, {r['state']}) | {r['stars']} stars from "
            f"{r['review_count']} reviews | price {price} | {r['categories']}"
        )
        for s in r["snippets"]:
            lines.append(f'   - "{s}"')
    return "\n".join(lines)


def generate_answer(query: str, restaurants: list[dict], llm_cfg: dict) -> Iterator[str]:
    """Yield the answer text in pieces, so the UI can stream it."""
    if not restaurants:
        yield "No restaurants matched those filters. Try widening the search."
        return

    provider = llm_cfg.get("provider", "claude")
    prompt = build_prompt(query, restaurants)
    if provider == "claude":
        yield from _claude(prompt, llm_cfg)
    elif provider == "ollama":
        yield from _ollama(prompt, llm_cfg)
    elif provider != "none":
        raise ValueError(f"Unknown llm provider: {provider}")


def _claude(prompt: str, llm_cfg: dict) -> Iterator[str]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            "No Claude API key provided. Set the ANTHROPIC_API_KEY environment variable, "
            "or set llm.provider to none in config.yaml."
        )

    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    try:
        with client.messages.stream(
            model=llm_cfg["claude_model"],
            max_tokens=llm_cfg["max_tokens"],
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream
    except anthropic.AuthenticationError:
        yield "\n\n[Claude API key missing or invalid. Set ANTHROPIC_API_KEY and restart the app.]"
    except anthropic.APIConnectionError:
        yield "\n\n[Could not reach the Claude API. Check your internet connection.]"
    except anthropic.APIStatusError as e:
        yield f"\n\n[Claude API error {e.status_code}: {e.message}]"


def _ollama(prompt: str, llm_cfg: dict) -> Iterator[str]:
    import requests

    try:
        with requests.post(
            f"{llm_cfg['ollama_url'].rstrip('/')}/api/chat",
            json={
                "model": llm_cfg["ollama_model"],
                "stream": True,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("message", {}).get("content", "")
    except requests.RequestException as e:
        yield f"\n\n[Could not get an answer from Ollama at {llm_cfg['ollama_url']}: {e}]"
