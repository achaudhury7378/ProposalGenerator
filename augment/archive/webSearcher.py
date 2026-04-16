import os
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from autogen_ext.models.ollama import OllamaChatCompletionClient

llama_model_client = OllamaChatCompletionClient(
    model="llama3.2:3b",  # or "llama3.2" depending on your model name
    base_url="http://localhost:11434/api/generate",  # Optional if using default
    # timeout=60  # Optional timeout setting
)

# Initialize Tavily client
# (Set your API Key as TAVILY_API_KEY in env variable or replace below)
tavily_client = TavilyClient(api_key="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")

def tavily_search(topic, max_results=5):
    """Search blogs/articles using Tavily and return top result URLs."""
    response = tavily_client.search(topic)
    urls = []
    if 'results' in response:
        for item in response['results'][:max_results]:
            url = item.get('url')
            if url:
                urls.append(url)
    return urls

def scrape_blog_content(url):
    """Fetch and parse blog post content from the given url."""
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            print(f"Failed to fetch {url} (Status: {res.status_code})")
            return ""
        soup = BeautifulSoup(res.content, 'html.parser')
        # Try typical blog containers
        if soup.find('article'):
            return soup.find('article').get_text(separator=' ', strip=True)
        elif soup.find('main'):
            return soup.find('main').get_text(separator=' ', strip=True)
        elif soup.find('body'):
            return soup.find('body').get_text(separator=' ', strip=True)
        else:
            return ""
    except Exception as e:
        print(f"Exception scraping {url}: {e}")
        return ""

def aggregate_blog_contents(query: str, max_results: int = 1):
    """Search and scrape multiple blogs on a topic for later LLM use."""
    urls = tavily_search(query, max_results)
    contents = []
    for url in urls:
        print(f"Scraping: {url}")
        content = scrape_blog_content(url)
        if content:
            contents.append(content)
    combined_text = '\n\n'.join(contents)
    prompt = '''Please provide a comprehensive executive summary that synthesizes all the information into a clear, rich with info to write a proposal from it.'''
    return combined_text

# Example Usage
# if __name__ == "__main__":
#     topic = "Emerging trends in AI for 2025"
#     collected_content = aggregate_blog_contents(topic, max_results=5)
#     print(f"Aggregated blog content for '{topic}':\n")
#     print(collected_content)
    # Next: feed `collected_content` into your favorite LLM prompt

