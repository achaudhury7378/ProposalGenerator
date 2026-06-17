import asyncio
from augment.tools.get_links import tavily_search
from scrapegraphai.graphs import SmartScraperGraph
import os

tavily_key = os.getenv('TAVILY_API_KEY')

# ScrapegraphAI Ollama config — uses the local Ollama server
_SCRAPER_CONFIG = {
    "llm": {
        "model": "ollama/gemma4:26b",
        "temperature": 0.1,
        "base_url": "http://localhost:11434",
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",
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
    input_urls = await asyncio.to_thread(tavily_search, topic, tavily_key)
    tasks = [get_data(url, prompt) for url in input_urls]
    data_out = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in data_out if not isinstance(r, Exception)]

