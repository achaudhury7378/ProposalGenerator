from typing import Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import random
import openlit
from langfuse import Langfuse
import requests
from bs4 import BeautifulSoup
import re
from augment.archive.webSearcher import aggregate_blog_contents
with open("../../openapikey.txt", "r") as f:
    # print(f.readlines()[0])
    api_key = f.readlines()[0]

def extract_agent_responses(messages):
    """Extract responses from each agent based on their names"""
    agent_responses = {}

    for message in messages:
        if hasattr(message, 'source') and hasattr(message, 'content'):
            source = message.source
            content = message.content

            # Map agent names to your desired variable names
            if source == "MarketAnalyst":
                agent_responses['market_research_response'] = content
            elif source == "FinancialPlanner":
                agent_responses['financial_planner_response'] = content
            elif source == "SolutionDesigner":
                agent_responses['solution_architect_response'] = content
            elif source == "CustomerEngagement_Analyst":
                agent_responses['customer_researcher_response'] = content
            elif source == "RiskManager":
                agent_responses['risk_analyst_response'] = content
        print("Finalized Contents")
        print(agent_responses)

    return agent_responses


def extract_text_from_url(url: str) -> str:
    """
    Fetches the content from a URL and extracts the most meaningful readable text,
    including structured sections.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove common non-content elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript']):
            tag.decompose()

        # Extract headings & paragraphs for structure
        content_elements = soup.find_all(['h1','h2','h3','h4','h5','h6','p'])
        text_chunks = []
        for el in content_elements:
            text = el.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            if len(text) > 40:  # Skip very short or irrelevant lines
                text_chunks.append(text)

        return '\n\n'.join(text_chunks)
    except Exception as e:
        print(f"Failed to extract text from {url}: {e}")
        return ""

from autogen_core.tools import FunctionTool

tavily_search_tool = FunctionTool(
    aggregate_blog_contents,
    description="Perform a web search using Tavily, returning URLs and snippet content."
)
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",           # or "gpt-3.5-turbo", "gpt-4-turbo", etc.
    api_key=api_key,         # your real OpenAI API key
)
##################PROMPTS   ###############################################
import configparser

config = configparser.ConfigParser()
config.read('prompts.cfg')

##################TOOOOOOOOOOOOOOOOOOOOOOOOOOOOLSSSSSSSSSSSSSSSSSSSSSSSSS#################
# Note: This example uses mock tools instead of real APIs for demonstration purposes
import asyncio

        
        
####################################AGENTS##################################################
section_creater_agent = AssistantAgent(
    "SectionCreatorAgent",
    description="An agent for creating section for Proposal for given topic, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    Your role as a Sections Creator Agent is to break down the process of creating a proposal into well-defined sections. You will identify and list the essential sections required for a comprehensive and professional proposal. You will not execute or delegate tasks beyond creating these sections.

Instructions:
we need max of 7 sections
Identify and list all necessary sections required for a comprehensive proposal.
once you get response from you team member use it in proposal sections to create a professional proposal.
Ensure each section is logically ordered to create a cohesive and professional proposal. no need to give much details about sections.
    """,
)


planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the engage only once and  after getting  required sections come from SectionCreatorAgent ",
    model_client=model_client,
    system_message="""
    You are a planning agent. Your job is to assign agents to complete all the sections (received from SectionCreatorAgent) of the Proposal. Your team members are:

    MarketAnalyst: Use for researching industry trends, competition, and customer needs.
    FinancialPlanner: Use for developing financial forecasts, budgeting, and revenue models.
    SolutionDesigner: Use for defining product/service features and implementation roadmap.
    CustomerEngagementAnalyst: Use for identifying customer pain points and crafting value propositions.
    RiskManager: Use for identifying business risks, ensuring compliance, and developing mitigation plans.
    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    keep taking their findings,and fill all the section that was created by SectionCreaterAgent in a representable manner (considering we are shoing this tpropsal to higher management level) once all the team embers ae done and sections completed  then  end with  "TERMINATE" to stop te conversation

""",
)

#**************************PROFESHIONAL AGENTS **********************************
market_researcher = AssistantAgent(
        "MarketAnalyst",
        model_client,
        tools=[tavily_search_tool],
        description="Researches industry trends, competition, and customer needs to define market positioning and opportunities. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Market Analysis']['prompt']
    )

financial_researcher = AssistantAgent(
        "FinancialPlanner",
        model_client,
        tools=[tavily_search_tool],
        description="Develops financial forecasts, budgeting, and revenue models to ensure business viability and profitability.You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Financial Planning']['prompt']
    )
solution_architect = AssistantAgent(
        "SolutionDesigner",
        model_client,
        tools=[tavily_search_tool],
        description="Defines the product/service features, implementation roadmap, and ensures technical and operational feasibility. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Solution Design']['prompt']
    )

customer_researcher = AssistantAgent(
        "CustomerEngagement_Analyst",
        model_client,
        tools=[tavily_search_tool],
        description="Identifies customer pain points, crafts value propositions, and designs engagement strategies. You have access to a search tool, you can use it to enhance your search.",
        system_message=config['Client Engagement']['prompt']
    )

risk_analyst = AssistantAgent(
        "RiskManager",
        model_client,
        tools=[tavily_search_tool],
        description="Identifies business risks, ensures compliance, and develops mitigation and contingency plans. You have access to a search tool , you can use it to enhance your search.",
        system_message=config['Risk Management']['prompt']
    )
############TERMINATION CONDITION ##################################
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination
#############SELECTOR FUNCTION ################
def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
    print("len of msg",len(messages))
    if len(messages)==1:
        print("calling selector agent")
        return section_creater_agent.name
    if messages[-1].source != planning_agent.name:
        print("calling planning agent")
        return planning_agent.name
    return None

def selector_func_with_user_proxy(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
    print("len of msg",len(messages))
    if len(messages)==1:
        print("calling selector agent")
        return section_creater_agent.name
    if messages[-1].source != planning_agent.name and messages[-1].source != user_proxy_agent.name:
        print("calling planning agent")
        return planning_agent.name

    if messages[-1].source == planning_agent.name:
        if messages[-2].source == user_proxy_agent.name and "APPROVE" in messages[-1].content.upper():  # type: ignore
            # User has approved the plan, proceed to the next agent.
            return None
        # Use the user proxy agent to get the user's approval to proceed.
        return user_proxy_agent.name
    if messages[-1].source == user_proxy_agent.name:
        # If the user does not approve, return to the planning agent.
        if "APPROVE" not in messages[-1].content.upper():  # type: ignore
            return planning_agent.name
    return None
##############USER PROXY #########################################
user_proxy_agent = UserProxyAgent("UserProxyAgent", description="A proxy for the user to approve or disapprove tasks.")
############ SELECTOR PROMPT #######################################

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

############# RUNNING TEAM ###############################################
team = SelectorGroupChat(
    [section_creater_agent,planning_agent, market_researcher,financial_researcher,solution_architect,customer_researcher,risk_analyst,user_proxy_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
    selector_func=selector_func_with_user_proxy#selector_func,
)

################### TASK #################################################
task = f"""Query {random.randint(0,1000)}: can you write a proposal on topic:Inventory Management in Supply Chains  ? """
# Use asyncio.run(..i.) if you are running this in a script. 
async def main():
    await team.reset()
    task_result=await Console(team.run_stream(task=task))
    last_message = task_result.messages
    print("------------------")
    print(task_result.messages)
    print("------------------")
    agent_responses = extract_agent_responses(task_result.messages)
    print(agent_responses)

asyncio.run(main())

    
        