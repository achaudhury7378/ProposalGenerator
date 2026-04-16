import os
from tavily import TavilyClient



def tavily_search(topic,env_key, max_results=5):
    tavily_client = TavilyClient(api_key=env_key)
    """Search blogs/articles using Tavily and return top result URLs."""
    response = tavily_client.search("blogs regarding "+topic)
    urls = []
    if 'results' in response:
        for item in response['results'][:max_results]:
            url = item.get('url')
            if url:
                urls.append(url)
    return urls

# print(tavily_search("inventory management strategies"))