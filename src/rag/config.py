import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB Settings
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "alzheimer_articles")

# Model Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

# PubMed Settings
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

# Processing Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
