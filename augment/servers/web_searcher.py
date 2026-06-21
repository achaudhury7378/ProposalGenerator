import asyncio
from augment.tools.get_links import tavily_search
from scrapegraphai.graphs import SmartScraperGraph
import os
from augment.settings import OLLAMA_HOST, OLLAMA_MODEL, TAVILY_API_KEY,OLLAMA_EMBED_MODEL
tavily_key = TAVILY_API_KEY

# ScrapegraphAI Ollama config — uses the local Ollama server
_SCRAPER_CONFIG = {
    "llm": {
        "model": "ollama/"+OLLAMA_MODEL,
        "temperature": 0.1,
        "base_url": OLLAMA_HOST,
    },
    "embeddings": {
        "model": "ollama/"+OLLAMA_EMBED_MODEL,
        "base_url": OLLAMA_HOST,
    },
    "verbose": False,
}

async def main_researcher(topic, prompt):
    async def get_data(url, prompt):
        graph = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=_SCRAPER_CONFIG,
        )
        result = await asyncio.wait_for(asyncio.to_thread(graph.run),timeout=90)
        return result

    # 3. Run all URLs concurrently
    input_urls = await asyncio.to_thread(tavily_search, topic, TAVILY_API_KEY)
    tasks = [get_data(url, prompt) for url in input_urls]
    data_out = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in data_out if not isinstance(r, Exception)]

