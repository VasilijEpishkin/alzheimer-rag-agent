from typing import List, Tuple
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str) -> List[str]:
    # Simple character-based chunking for demonstration
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(text[i:i + CHUNK_SIZE])
    return chunks

def smart_chunk_with_overlap(tokens: List[int], chunk_tokens: int, overlap_tokens: int, tokenizer) -> List[Tuple[int, int, str]]:
    segments = []
    start = 0
    total_len = len(tokens)
    
    while start < total_len:
        end = min(start + chunk_tokens, total_len)
        
        # Heuristic to not break in the middle of a sentence
        if end < total_len:
            window_text = tokenizer.decode(tokens[start:end])
            last_period = window_text.rfind('.')
            
            if last_period != -1 and last_period > len(window_text) * 0.5:
                clean_text = window_text[:last_period+1]
                clean_ids = tokenizer.encode(clean_text, add_special_tokens=False)
                new_end = start + len(clean_ids)
                if new_end > start:
                    end = new_end
        
        segment_ids = tokens[start:end]
        segment_text = tokenizer.decode(segment_ids)
        segments.append((start, end, segment_text))
        
        if end >= total_len:
            break
        start = end - overlap_tokens
        if start < 0: start = 0
            
    return segments
