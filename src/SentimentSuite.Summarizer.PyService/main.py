from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging

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

def safe_chunk_text(text, max_words=400):
    """Split text into safe chunks for BART (much smaller chunks)"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk)
    
    return chunks

@app.post("/summarize")
async def summarize(req: SummaryRequest):
    try:
        if not summarizer:
            raise HTTPException(status_code=500, detail="Summarizer not initialized")
        
        logger.info(f"Processing text of length: {len(req.text)}")
        
        if len(req.text.strip()) < 10:
            return {"summary": "Text too short to summarize"}
        
        # Always chunk for safety - BART has strict limits
        chunks = safe_chunk_text(req.text, 300)  # Very conservative chunk size
        logger.info(f"Split into {len(chunks)} chunks")
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)})")
                summary = summarizer(
                    chunk,
                    max_length=60,  # Much smaller output
                    min_length=20,
                    do_sample=False
                )
                chunk_summaries.append(summary[0]['summary_text'])
                logger.info(f"Successfully processed chunk {i+1}")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Skip failed chunks instead of crashing
                continue
        
        if not chunk_summaries:
            return {"summary": "Failed to process any chunks"}
        
        # Simply join the summaries
        final_summary = ' '.join(chunk_summaries)
        logger.info("Summary generation completed")
        return {"summary": final_summary}
        
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")