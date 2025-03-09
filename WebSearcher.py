#create a websearch aent using autogen
from duckduckgo_search import DDGS
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
import configparser

config = configparser.ConfigParser()
config.read('prompts.cfg')
print(config["Market Analysis"]["prompt"])
model_client = OpenAIChatCompletionClient(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)


async def main() -> None:
    async def search(query:str) -> str:
        '''
        This function uses the DuckDuckGo search API to search for the query and returns the results.
        '''
        results = DDGS().chat(query,model='llama-3.3-70b')
        print("----------------")
        print(query)
        print("----------------")
        return results

    market_researcher = AssistantAgent(
        "Market_Analyst",
        model_client,
        tools=[search],
        description="Researches industry trends, competition, and customer needs to define market positioning and opportunities. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Market Analysis']['prompt']
    )
    financial_researcher = AssistantAgent(
        "Financial_Planner",
        model_client,
        tools=[search],
        description="Develops financial forecasts, budgeting, and revenue models to ensure business viability and profitability. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Financial Planning']['prompt']
    )
    solution_architect = AssistantAgent(
        "Solution_Designer",
        model_client,
        tools=[search],
        description="Defines the product/service features, implementation roadmap, and ensures technical and operational feasibility. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Solution Design']['prompt']
    )
    customer_researcher = AssistantAgent(
        "Customer_Engagement_Analyst",
        model_client,
        tools=[search],
        description="Identifies customer pain points, crafts value propositions, and designs engagement strategies. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Client Engagement']['prompt']
    )
    risk_analyst = AssistantAgent(
        "Risk_Manager",
        model_client,
        tools=[search],
        description="Identifies business risks, ensures compliance, and develops mitigation and contingency plans. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Risk Management']['prompt']
    )
    termination = TextMentionTermination("TERMINATE")
    team = SelectorGroupChat(
        [market_researcher, financial_researcher, solution_architect, customer_researcher, risk_analyst],
        model_client=model_client,
        termination_condition=termination,
    )
    await Console(team.run_stream(task="Client is facing problems in inventory management and wants to improve their supply chain."))


asyncio.run(main())