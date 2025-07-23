from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging
import asyncio
import concurrent.futures
from typing import List

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    logger.info("Summarizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load summarizer: {e}")
    summarizer = None

class SummaryRequest(BaseModel):
    text: str

def smart_chunk_text(text: str, max_tokens: int = 512, overlap_sentences: int = 2) -> List[str]:
    """Split text into semantic chunks using sentence boundaries with overlap"""
    # Split into sentences more carefully
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if len(sentences) <= 3:
        return [text]  # Too short to chunk meaningfully
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = len(sentence.split())
        
        # If adding this sentence exceeds limit and we have content, finalize chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Keep overlap from previous chunk
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_tokens = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_single_chunk(chunk: str, chunk_id: int) -> tuple[int, str]:
    """Process a single chunk and return (id, summary) for ordering"""
    try:
        logger.info(f"Processing chunk {chunk_id} (length: {len(chunk)} chars)")
        
        summary = summarizer(
            chunk,
            max_length=80,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        
        result = summary[0]['summary_text']
        logger.info(f"Successfully processed chunk {chunk_id}")
        return chunk_id, result
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {e}")
        return chunk_id, f"[Chunk {chunk_id} failed to process]"

async def parallel_summarize_chunks(chunks: List[str]) -> List[str]:
    """Process chunks in parallel using ThreadPoolExecutor"""
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        tasks = [
            loop.run_in_executor(executor, process_single_chunk, chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
    
    # Sort by chunk_id to maintain order
    results.sort(key=lambda x: x[0])
    return [summary for _, summary in results]

def create_final_summary(chunk_summaries: List[str]) -> str:
    """Create coherent final summary from chunk summaries"""
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    # Join summaries and create meta-summary if chunks are substantial
    combined = ' '.join(chunk_summaries)
    
    # If combined summary is still long, run another summarization pass
    if len(combined.split()) > 150:
        try:
            logger.info("Creating meta-summary from chunk summaries")
            meta_summary = summarizer(
                combined,
                max_length=100,
                min_length=40,
                do_sample=False,
                truncation=True
            )
            return meta_summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Meta-summary failed: {e}, returning concatenated")
            return combined
    
    return combined

@app.post("/summarize")
async def summarize(req: SummaryRequest):
    try:
        if not summarizer:
            raise HTTPException(status_code=500, detail="Summarizer not initialized")
        
        text = req.text.strip()
        logger.info(f"Processing text of length: {len(text)} chars, {len(text.split())} words")
        
        if len(text) < 50:
            return {"summary": "Text too short to summarize"}
        
        # Smart chunking with sentence boundaries
        chunks = smart_chunk_text(text, max_tokens=600, overlap_sentences=2)
        logger.info(f"Split into {len(chunks)} semantic chunks")
        
        if len(chunks) == 1:
            # Single chunk - process directly
            summary = summarizer(
                chunks[0],
                max_length=120,
                min_length=40,
                do_sample=False,
                truncation=True
            )
            final_summary = summary[0]['summary_text']
        else:
            # Multiple chunks - parallel processing + hierarchical summarization
            chunk_summaries = await parallel_summarize_chunks(chunks)
            
            # Filter out failed chunks
            valid_summaries = [s for s in chunk_summaries if not s.startswith("[Chunk")]
            
            if not valid_summaries:
                return {"summary": "Failed to process any chunks"}
            
            final_summary = create_final_summary(valid_summaries)
        
        logger.info("Summary generation completed")
        return {"summary": final_summary}
        
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": summarizer is not None}