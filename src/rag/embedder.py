from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL

def get_embedder():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
