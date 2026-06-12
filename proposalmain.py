from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, MagenticOrchestration, StandardMagenticManager
from semantic_kernel.functions import kernel_function
from augment.servers.web_searcher import main_researcher
from augment.tools.research_tool import deep_research
import os
import asyncio
import configparser

from dotenv import load_dotenv

load_dotenv()

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")

config = configparser.RawConfigParser()
config.read('prompts.cfg')


class ResearchPlugin:
    @kernel_function(name="web_search", description="Search and scrape the web for a topic")
    async def web_search(self, topic: str, prompt: str) -> str:
        results = await main_researcher(topic, prompt)
        return "\n\n".join(str(r) for r in results if r)

    @kernel_function(name="deep_research", description="Deep multi-query research on a topic")
    async def deep_research_fn(self, topic: str) -> str:
        return await deep_research(topic)


def _make_kernel(with_tools: bool = False) -> Kernel:
    kernel = Kernel()
    kernel.add_service(OllamaChatCompletion(
        ai_model_id=_OLLAMA_MODEL,
        host=_OLLAMA_HOST,
    ))
    if with_tools:
        kernel.add_plugin(ResearchPlugin(), plugin_name="Research")
    return kernel


main_agent = ChatCompletionAgent(
    "main",
    instruction=config['Main Agent']['prompt'],
    kernel=_make_kernel(),
)
synthesizer = ChatCompletionAgent(
    "synthesizer",
    instruction=config["Combiner"]['prompt'],
    kernel=_make_kernel(),
)

market_researcher = ChatCompletionAgent(
    "MarketAnalyst",
    instruction=config['Market Analysis']['prompt'],
    description=config['Market Analysis']['description'],
    kernel=_make_kernel(with_tools=True),
)

financial_researcher = ChatCompletionAgent(
    "FinancialPlanner",
    instruction=config['Financial Planning']['prompt'],
    description=config['Financial Planning']['description'],
    kernel=_make_kernel(with_tools=True),
)

solution_architect = ChatCompletionAgent(
    "SolutionDesigner",
    instruction=config['Solution Design']['prompt'],
    description=config['Solution Design']['description'],
    kernel=_make_kernel(with_tools=True),
)

customer_researcher = ChatCompletionAgent(
    "CustomerEngagement_Analyst",
    instruction=config['Client Engagement']['prompt'],
    description=config['Client Engagement']['description'],
    kernel=_make_kernel(with_tools=True),
)

risk_analyst = ChatCompletionAgent(
    "RiskManager",
    instruction=config['Risk Management']['prompt'],
    description=config['Risk Management']['description'],
    kernel=_make_kernel(with_tools=True),
)

manager = StandardMagenticManager(
    agents=[
        main_agent,
        synthesizer,
        market_researcher,
        financial_researcher,
        solution_architect,
        customer_researcher,
        risk_analyst,
    ]
)

orchestration = MagenticOrchestration(manager=manager)


async def main():
    result = await orchestration.run(
        input="Create a proposal for supply chain optimization system"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
