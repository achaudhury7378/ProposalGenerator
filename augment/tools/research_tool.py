import asyncio
import os
from typing import List

from openai import AsyncOpenAI
from augment.tools.get_links import tavily_search
from augment.servers.web_searcher import main_researcher

# Ollama runs an OpenAI-compatible API on localhost
_ollama = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't validate the key
)
_model = "gemma4:26b"
_tavily_key = os.getenv("TAVILY_API_KEY")


async def _generate_queries(topic: str, num_queries: int) -> List[str]:
    """Ask the local Ollama model to produce diverse sub-queries for a topic."""
    resp = await _ollama.chat.completions.create(
        model=_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research strategist. Given a topic, generate distinct "
                    "search queries that together give comprehensive coverage from "
                    "different angles: overview, market data, technical details, "
                    "challenges, and recent news. Return one query per line, no numbering."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {topic}\nGenerate {num_queries} search queries.",
            },
        ],
        temperature=0.4,
    )
    raw = resp.choices[0].message.content.strip().splitlines()
    return [q.strip() for q in raw if q.strip()][:num_queries]


async def _scrape_query(query: str, scrape_prompt: str) -> List[str]:
    """Search via Tavily then scrape results; returns a list of scraped strings."""
    try:
        urls = await asyncio.to_thread(tavily_search, query, _tavily_key)
        if not urls:
            return []
        results = await main_researcher(query, scrape_prompt)
        return [str(r) for r in results if r]
    except Exception as exc:
        return [f"[error on query '{query}': {exc}]"]


async def _synthesize(topic: str, chunks: List[str]) -> str:
    """Synthesize scraped chunks into a structured report via the local Ollama model."""
    combined = "\n\n---SOURCE BREAK---\n\n".join(chunks[:12])  # keep context manageable

    resp = await _ollama.chat.completions.create(
        model=_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior research analyst. Synthesize the raw web data "
                    "below into a structured, fact-rich report with these sections: "
                    "Overview, Key Findings, Market / Technical Details, Challenges, "
                    "Recent Developments. Cite facts precisely; do not invent numbers."
                ),
            },
            {
                "role": "user",
                "content": f"Research topic: {topic}\n\nRaw data:\n{combined}",
            },
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


async def deep_research(topic: str, num_queries: int = 4) -> str:
    """
    Deep-research agent powered by local Ollama (gemma4:26b).

    Steps:
      1. Generate multiple diverse search queries via Ollama.
      2. For each query: search with Tavily, scrape URLs with ScrapegraphAI (Ollama).
      3. Synthesize all scraped content into a structured report with Ollama.

    Args:
        topic:       The subject to research.
        num_queries: Number of distinct search angles to explore (default 4).

    Returns:
        A synthesised research report as a plain string.
    """
    scrape_prompt = (
        "Extract all relevant, detailed information about the topic from this page. "
        "Include specific data points, statistics, names, dates, and direct quotes."
    )

    try:
        # 1. Generate diverse sub-queries
        queries = await _generate_queries(topic, num_queries)

        # 2. Scrape all queries concurrently
        tasks = [_scrape_query(q, scrape_prompt) for q in queries]
        nested = await asyncio.gather(*tasks)

        chunks: List[str] = []
        for batch in nested:
            chunks.extend(batch)

        if not chunks:
            return f"No research results found for: {topic}"

        # 3. Synthesize into a coherent report
        report = await _synthesize(topic, chunks)

        header = (
            f"## Deep Research Report: {topic}\n"
            f"_Model: {_model} | Queries: {len(queries)} | Sources: {len(chunks)}_\n\n"
        )
        return header + report

    except Exception as exc:
        return f"Deep research failed for '{topic}': {exc}"
