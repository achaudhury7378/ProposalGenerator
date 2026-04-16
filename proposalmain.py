from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, MagenticOrchestration, StandardMagenticManager,OpenAIAssistantAgent
from augment.servers.web_searcher import main_reasecher as tavily_search_tool, api_key
import os
import asyncio
api_key = os.getenv("OPENAI_API_KEY")

# Kernel for o3-mini (main + synthesizer)
o3_kernel = Kernel()
o3_service = OpenAIChatCompletion(
    ai_model_id="o3-mini",
    api_key=api_key
)
o3_kernel.add_service(o3_service)

# Kernel for gpt-4o-mini (research agents)
gpt4o_kernel = Kernel()
gpt4o_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    api_key=api_key
)
gpt4o_kernel.add_service(gpt4o_service)

################## PROMPTS   ###############################################
import configparser

config = configparser.ConfigParser()
config.read('prompts.cfg')

##################################### AGENTS ####################################################

        
        
#################################### MAIN AGENTS##################################################
main_agent = ChatCompletionAgent("main", instruction = config['Main Agent']['prompt'] , kernel = o3_kernel)
synthesizer = ChatCompletionAgent("synthesizer", instruction = config["Combiner"]['prompt'], kernel = gpt4o_kernel)
#**************************PROFESSIONAL AGENTS **********************************
market_researcher = gpt4o_kernel.agent_framework.openai_assistant_agent(
        "MarketAnalyst",
        description=config['Market Analysis']['description'],
        system_message=config['Market Analysis']['prompt'],
        kernel = gpt4o_kernel
    )

financial_researcher = gpt4o_kernel.agent_framework.openai_assistant_agent(
        "FinancialPlanner",
        tools=[tavily_search_tool],
        description=config['Financial Planning']['description'],
        system_message=config['Financial Planning']['prompt'],
        kernel = gpt4o_kernel
    )
solution_architect = gpt4o_kernel.agent_framework.openai_assistant_agent(
        "SolutionDesigner",
        tools=[tavily_search_tool],
        description=config['Solution Design']['description'],
        system_message=config['Solution Design']['prompt'],
        kernel = gpt4o_kernel
    )

customer_researcher = gpt4o_kernel.agent_framework.openai_assistant_agent(
        "CustomerEngagement_Analyst",
        tools=[tavily_search_tool],
        description=config['Client Engagement']['description'],
        system_message=config['Client Engagement']['prompt'],
        kernel = gpt4o_kernel
    )

risk_analyst = gpt4o_kernel.agent_framework.openai_assistant_agent(
        "RiskManager",
        tools=[tavily_search_tool],
        description=config['Risk Management']['description'],
        system_message=config['Risk Management']['prompt'],
        kernel = gpt4o_kernel
    )

manager = StandardMagenticManager(
    agents=[
        main_agent,
        synthesizer,
        market_researcher,
        financial_researcher,
        solution_architect,
        customer_researcher,
        risk_analyst
    ]
)


orchestration = MagenticOrchestration(
    manager=manager
)


async def main():

    result = await orchestration.run(
        input="Create a proposal for supply chain optimization system"
    )

    print(result)

asyncio.run(main())


    
        