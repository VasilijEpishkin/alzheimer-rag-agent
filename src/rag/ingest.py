import pandas as pd
from rag.store import VectorStore
from rag.utils import chunk_text
from rag.logger import logger

class Ingestor:
    def __init__(self):
        self.store = VectorStore()

    def run(self, csv_path: str):
        df = pd.read_csv(csv_path)
        logger.info(f"Ingesting {len(df)} articles")
        
        all_chunks = []
        all_metadatas = []
        
        for _, row in df.iterrows():
            text = f"{row['title']}\n\n{row['abstract']}"
            chunks = chunk_text(text)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    "pmid": row["pmid"],
                    "title": row["title"]
                })
        
        self.store.add_documents(all_chunks, all_metadatas)
        logger.info("Ingestion complete")
