import asyncio
import configparser
from augment.tools.get_links import tavily_search
from scrapegraphai.graphs import SmartScraperGraph
import os

config = configparser.ConfigParser()
config.read('prompts.cfg')
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
        result = graph.run()
        return result

    # 3. Run all URLs concurrently
    input_urls = tavily_search(topic, tavily_key)
    tasks = [get_data(url, prompt) for url in input_urls]
    data_out = await asyncio.gather(*tasks)
    return data_out

async def market_researcher(topic):
    return await main_researcher(topic, config["Market Analysis"]["crawler_prompt"])

async def financial_researcher(topic):
    return await main_researcher(topic, config["Financial Planning"]["crawler_prompt"])

async def solution_researcher(topic):
    return await main_researcher(topic, config["Solution Design"]["crawler_prompt"])

async def customer_researcher(topic):
    return await main_researcher(topic, config["Client Engagement"]["crawler_prompt"])

async def risk_researcher(topic):
    return await main_researcher(topic, config["Risk Management"]["crawler_prompt"])

if __name__ == "__main__":
    topic = "problems in inventory management"
    # For testing purposes, we use asyncio.run
    data = asyncio.run(market_researcher(topic))
    print("Complete")
    print(data)
    print(len(data))
