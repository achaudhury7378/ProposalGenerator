import os
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from autogen_ext.models.ollama import OllamaChatCompletionClient


# Initialize Tavily client
# (Set your API Key as TAVILY_API_KEY in env variable or replace below)
tavily_client = TavilyClient(api_key="***REMOVED***")

def tavily_search(topic, max_results=5):
    """Search blogs/articles using Tavily and return top result URLs."""
    response = tavily_client.search("blogs regarding "+topic)
    urls = []
    if 'results' in response:
        for item in response['results'][:max_results]:
            url = item.get('url')
            if url:
                urls.append(url)
    return urls