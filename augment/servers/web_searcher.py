import asyncio
import configparser
from augment.tools.get_links import tavily_search
from scrapegraphai.graphs import SmartScraperGraph
import os
config = configparser.ConfigParser()
config.read('/Users/abhisheknarayanchaudhury/Documents/Codes/Proposal Generator/prompts.cfg')
api_key = os.getenv('OPENAI_API_KEY')
tavily_key = os.getenv('TAVILY_API_KEY')
async def main_reasecher(topic,prompt):

    async def get_data(url,prompt):

        graph = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config={
                "llm": {
                    "model": "gpt-4.1-nano",  # or gpt-4o, gpt-4-turbo
                    "api_key": api_key,
                    "temperature": 0.1
                },
                "max_tokens": 3000
            }
        )
        result = graph.run()
        return result

    # 3. Run all URLs concurrently
    input_urls = tavily_search(topic,tavily_key)
    tasks = [get_data(url,prompt) for url in input_urls]
    data_out = await asyncio.gather(*tasks)
    return data_out
def market_reseacher(topic):
    data = asyncio.run(main_reasecher(topic,config["Market Analysis"]["crawler_prompt"]))
    return data
def financial_researcher(topic):
    data = asyncio.run(main_reasecher(topic,config["Financial Planning"]["crawler_prompt"]))
    return data
def solution_researcher(topic):
    data = asyncio.run(main_reasecher(topic,config["Solution Design"]["crawler_prompt"]))
    return data
def customer_researcher(topic):
    data = asyncio.run(main_reasecher(topic,config["Client Engagement"]["crawler_prompt"]))
    return data
def risk_researcher(topic):
    data = asyncio.run(main_reasecher(topic,config["Risk Management"]["crawler_prompt"]))
    return data

if __name__ == "__main__":
    topic = "problems in inventory management"
    data = market_reseacher(topic)
    print("Complete")
    print(data)
    print(len(data))

