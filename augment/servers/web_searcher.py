import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai import LLMExtractionStrategy
import configparser

config = configparser.ConfigParser()
config.read('/Users/abhisheknarayanchaudhury/Documents/Codes/Proposal Generator/prompts.cfg')
print(config["Crawler"]["prompt"])

class Product(BaseModel):
    name: str
    price: str
with open("/Users/abhisheknarayanchaudhury/Documents/Codes/Proposal Generator/openapikey.txt","r") as f:
    f.readlines()
    api_key = f.readlines()
async def main(input_urls):
    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(provider="openai/gpt-4.1-nano", api_token="***REMOVED***"),
        schema=Product.model_json_schema(), # Or use model_json_schema()
        extraction_type="schema",
        instruction=config["Crawler"]["prompt"],
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",   # or "html", "fit_markdown"
        extra_args={"temperature": 0.0, "max_tokens": 800}
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)
    # data_out = ""

    async def get_data(x):
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            # 4. Let's say we want to crawl a single page
            result = await crawler.arun(
                url=x,
                config=crawl_config
            )

            if result.success:
                # 5. The extracted content is presumably JSON
                data = json.loads(result.extracted_content)
                print("Extracted items:", data)

                # 6. Show usage stats
                llm_strategy.show_usage()  # prints token usage
            else:
                print("Error:", result.error_message)

    # 3. Run all URLs concurrently
    tasks = [get_data(url) for url in input_urls]
    data_out = await asyncio.gather(*tasks)
    return data_out

if __name__ == "__main__":
    urls = ["https://www.investopedia.com/terms/i/inventory-management.asp"]
    data = asyncio.run(main(urls))
    print(data)

