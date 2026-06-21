import os
from dotenv import load_dotenv

load_dotenv()   # load .env BEFORE anything reads os.getenv

OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_BASE_URL    = f"{OLLAMA_HOST}/v1"     # OpenAI-compatible endpoint
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")
