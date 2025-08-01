from typing import Sequence
import autogen_agentchat.agents
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient,AzureOpenAIChatCompletionClient
import asyncio
####################
# Assuming the DDGS class is already defined as provided
import requests
from bs4 import BeautifulSoup

# Assuming the DDGS class is already defined as provided  
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import os
from tavily import TavilyClient

def tavily_search(query: str, max_results: int = 5) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query, max_results=max_results)
    lines = []
    for result in response.get("results", []):
        lines.append(f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n")
    return "\n".join(lines)

from autogen_core.tools import FunctionTool

tavily_search_tool = FunctionTool(
    tavily_search,
    description="Perform a web search using Tavily, returning URLs and snippet content."
)




# async def get_links(keywords:str):
#     import requests
#     from bs4 import BeautifulSoup
#     web_query = keywords.replace(' ','+')
#
#     url = f"https://www.bing.com/search?q={web_query}"
#
#     payload = {}
#     headers = {
#         'Cookie': 'MUID=1D542C2CF0276D6124B639D1F1CC6CA9; SRCHD=AF=NOFORM; SRCHHPGUSR=SRCHLANG=en&IG=AB99B3DB0DBF49A485324C09043526F9; SRCHUID=V=2&GUID=52B9E6A40D474A6BA787E8CC27D0180D&dmnchg=1; SRCHUSR=DOB=20250531; _EDGE_S=F=1&SID=3CEA1EEFEC8B6713001D0B12ED606645; _EDGE_V=1; _SS=SID=3CEA1EEFEC8B6713001D0B12ED606645; ak_bmsc=4C8F4128975B846E6C41E7F22792FFFE~000000000000000000000000000000~YAAQr/TfF3nguMWWAQAAHsPEJBsh1qTLZZxINCGrnWs7OmqHIrYgDgY/PeLMvexijrYAX9Zw630WMrEQDftWLqxOTtzCS/QexNriYMY6cOM+1DnCvm3j8c6kycVhuyCZoTXXUm1gL1md991ef+GFifVzNRTnhNN41woODIB1SeUKxN0sht6Q2Z5c4U6zfgXmwKseUkxdPP7aylEDRqm7i4PCIu5w9444CKUHbB4R/5I1Fnk00+4NXlVlYMH8nuojgfZ0bNuCSlO/A9du9i8lGKybd1M/0vAq8fRtN3h7oCv6FdTHnk496Izrt1nvP7Ltpg3ZFw9q8xtGWNIYdE+KFH/zVGjpsUZG2zI=; MUIDB=1D542C2CF0276D6124B639D1F1CC6CA9'
#     }
#
#     response = requests.request("GET", url, headers=headers, data=payload)
#
#     html = response.text
#     soup = BeautifulSoup(html, 'html.parser')
#
#     divs = soup.find_all('div', class_='b_tpcn')
#     content_dict = ""
#     for div in divs:
#         link = div.find('a')
#         if link and link.has_attr('href'):
#             print(link['href'])
#             content = extract_text_from_url(link['href'])
#             content_dict += content
#
#
#     return f"these are search results for the {keywords} search result: {content_dict}"
#
# # Assuming the DDGS class is already defined as provided
#
# def extract_text_from_url(url: str) -> str:
#     """Fetch the content from the URL and extract the text."""
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Check if the request was successful
#         soup = BeautifulSoup(response.content, 'html.parser')
#
#         # Extract text from the page
#         paragraphs = soup.find_all('p')
#         page_text = ' '.join([para.get_text() for para in paragraphs])
#         return page_text.strip()
#     except Exception as e:
#         print(f"Failed to extract text from {url}: {e}")
#         return ""
        
# async def search_duckduckgo(keywords: str):
#     #if 3==3:
#     #    return "thanks unable to search"
#     with DDGS() as ddgs:
#         # Perform a text search
#         results = ddgs.text(keywords)
#
#         # Collect content from each URL
#         content_dict = {}
#         for result in results[:1]:
#             title = result['title']
#             url = result['href']
#             print(f"Fetching content from: {title} - {url}")
#             content = extract_text_from_url(url)
#             content_dict[title] = content
#             first2pairs = {k:content_dict[k] for k in list(content_dict)[:1]}
#             first2pairs=content_summerize(str(first2pairs))#####summerize with gpt3.5
#             print(first2pairs)
#         return first2pairs

def save_to_pdf(content_dict, filename="output.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for title, content in content_dict.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, title)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        pdf.ln(10)

    pdf.output(filename)
def save_to_txt(content_dict, filename="output.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        for title, content in content_dict.items():
            file.write(f"Title: {title}\n\n")
            file.write(f"{content}\n\n")
            file.write("="*80 + "\n\n")
    print(f"Content saved to {filename}")

#########################
import os
# from openai import AzureOpenAI
#
# endpoint = ""
# model_name = "gpt-35-turbo"
# deployment = ""
#
# subscription_key = "5"
# api_version = ""
# def content_summerize(query_content):
#     client = AzureOpenAI(
#     api_version=api_version,
#     azure_endpoint=endpoint,
#     api_key=subscription_key,
#     )
#     response = client.chat.completions.create(
#               messages=[
#              {
#                  "role": "system",
#                   "content": "summerize user input in 2-3 lines",
#               },
#               {
#                   "role": "user",
#                    "content": query_content,
#                }
#                       ],
#     max_tokens=4096,
#     temperature=1.0,
#     top_p=1.0,
#     model=deployment
#                    )
#    # print(response.choices[0].message.content)
#     return response.choices[0].message.content


    