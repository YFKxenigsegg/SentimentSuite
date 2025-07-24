import asyncio
import time
from typing import List, Dict, Any, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from core.interfaces import ISummarizer, TextChunk, SummaryResult, SummaryQuality
from core.config import settings, get_model_device, get_model_dtype

class OptimizedSummarizer(ISummarizer):
    """High-performance summarizer with batch processing and optimization"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = get_model_device()
        self.dtype = get_model_dtype()
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self.is_initialized = False
        
        # Quality presets
        self.quality_presets = {
            SummaryQuality.FAST: {
                "max_length": 60,
                "min_length": 20,
                "num_beams": 2,
                "do_sample": False,
                "early_stopping": True
            },
            SummaryQuality.BALANCED: {
                "max_length": 100,
                "min_length": 30,
                "num_beams": 4,
                "do_sample": False,
                "early_stopping": True
            },
            SummaryQuality.QUALITY: {
                "max_length": 150,
                "min_length": 40,
                "num_beams": 6,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "early_stopping": True
            }
        }
    
    async def initialize(self):
        """Initialize the model with optimizations"""
        if self.is_initialized:
            return
            
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self.is_initialized = True
    
    def _load_model(self):
        """Load and optimize the model"""
        try:
            # Load tokenizer and model separately for better control
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                cache_dir=settings.model_cache_dir
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.model_name,
                cache_dir=settings.model_cache_dir,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device == "cpu" or not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            # Apply optimizations
            self._apply_optimizations()
            
            # Create pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _apply_optimizations(self):
        """Apply various model optimizations"""
        if self.model is None:
            return
            
        # Enable evaluation mode
        self.model.eval()
        
        # Quantization (if enabled and supported)
        if settings.enable_quantization and hasattr(torch, 'quantization'):
            try:
                # Apply dynamic quantization
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"Quantization failed: {e}")
        
        # Compile model for PyTorch 2.0+ (if available)
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"Model compilation failed: {e}")
    
    async def summarize_chunk(
        self, 
        chunk: TextChunk, 
        quality: SummaryQuality = SummaryQuality.BALANCED
    ) -> SummaryResult:
        """Summarize a single chunk with quality control"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Get quality parameters
            params = self.quality_presets[quality].copy()
            
            # Adjust parameters based on chunk complexity
            params = await self._adapt_parameters_to_chunk(chunk, params)
            
            # Perform summarization
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._summarize_single,
                chunk.content,
                params
            )
            
            processing_time = time.time() - start_time
            
            # Calculate confidence score
            confidence = await self._calculate_confidence_score(
                chunk.content, result, chunk.metadata.complexity_score
            )
            
            return SummaryResult(
                content=result,
                confidence_score=confidence,
                processing_time=processing_time,
                chunk_id=chunk.metadata.id,
                metadata={
                    "quality": quality.value,
                    "chunk_complexity": chunk.metadata.complexity_score,
                    "parameters_used": params
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return SummaryResult(
                content=f"[Error processing chunk {chunk.metadata.id}: {str(e)}]",
                confidence_score=0.0,
                processing_time=processing_time,
                chunk_id=chunk.metadata.id,
                metadata={"error": str(e)}
            )
    
    async def batch_summarize(
        self, 
        chunks: List[TextChunk], 
        quality: SummaryQuality = SummaryQuality.BALANCED
    ) -> List[SummaryResult]:
        """Batch process multiple chunks with dynamic batching"""
        if not self.is_initialized:
            await self.initialize()
        
        if not chunks:
            return []
        
        # Group chunks into optimal batches
        batches = await self._create_optimal_batches(chunks)
        
        # Process batches in parallel with limited concurrency
        semaphore = asyncio.Semaphore(settings.max_workers)
        
        async def process_batch(batch_chunks):
            async with semaphore:
                return await self._process_batch(batch_chunks, quality)
        
        # Execute all batches
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Flatten results and sort by chunk ID
        all_results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                continue  # Skip failed batches
            all_results.extend(batch_result)
        
        # Sort by original chunk order
        all_results.sort(key=lambda x: x.chunk_id or 0)
        return all_results
    
    def is_ready(self) -> bool:
        """Check if summarizer is ready"""
        return self.is_initialized and self.pipeline is not None
    
    async def _adapt_parameters_to_chunk(
        self, 
        chunk: TextChunk, 
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt summarization parameters based on chunk characteristics"""
        params = base_params.copy()
        
        # Adjust length based on input size
        input_length = len(chunk.content.split())
        
        if input_length < 100:
            # Short chunks need shorter summaries
            params["max_length"] = min(params["max_length"], input_length // 2)
            params["min_length"] = min(params["min_length"], input_length // 4)
        elif input_length > 400:
            # Long chunks can have longer summaries
            params["max_length"] = min(params["max_length"] + 20, 200)
        
        # Adjust based on complexity
        if chunk.metadata.complexity_score > 0.7:
            # Complex content may need more beams for better quality
            params["num_beams"] = min(params.get("num_beams", 4) + 1, 8)
        elif chunk.metadata.complexity_score < 0.3:
            # Simple content can use fewer beams for speed
            params["num_beams"] = max(params.get("num_beams", 4) - 1, 2)
        
        return params
    
    def _summarize_single(self, text: str, params: Dict[str, Any]) -> str:
        """Perform single summarization (runs in thread pool)"""
        try:
            # Truncate input if too long
            max_input_length = 1024  # Adjust based on model
            if len(text.split()) > max_input_length:
                text = " ".join(text.split()[:max_input_length])
            
            result = self.pipeline(text, **params)
            return result[0]['summary_text']
            
        except Exception as e:
            raise RuntimeError(f"Summarization failed: {e}")
    
    async def _create_optimal_batches(self, chunks: List[TextChunk]) -> List[List[TextChunk]]:
        """Create optimal batches based on chunk size and complexity"""
        if not chunks:
            return []
        
        batches = []
        current_batch = []
        current_batch_tokens = 0
        max_batch_tokens = settings.batch_size * 300  # Approximate tokens per batch
        
        # Sort chunks by complexity (process similar complexity together)
        sorted_chunks = sorted(chunks, key=lambda x: x.metadata.complexity_score)
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.metadata.token_count
            
            # Check if adding this chunk would exceed batch limits
            if (len(current_batch) >= settings.batch_size or 
                current_batch_tokens + chunk_tokens > max_batch_tokens) and current_batch:
                
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0
            
            current_batch.append(chunk)
            current_batch_tokens += chunk_tokens
        
        # Add remaining chunks
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _process_batch(
        self, 
        batch_chunks: List[TextChunk], 
        quality: SummaryQuality
    ) -> List[SummaryResult]:
        """Process a batch of chunks"""
        # For now, process individually within batch
        # Could be optimized further with true batch processing
        tasks = [
            self.summarize_chunk(chunk, quality) 
            for chunk in batch_chunks
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _calculate_confidence_score(
        self, 
        original: str, 
        summary: str, 
        complexity: float
    ) -> float:
        """Calculate confidence score for the summary"""
        try:
            # Basic heuristics for confidence scoring
            original_length = len(original.split())
            summary_length = len(summary.split())
            
            # Compression ratio (should be reasonable)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            compression_score = 1.0 - abs(compression_ratio - 0.25)  # Target ~25% compression
            
            # Length appropriateness
            length_score = 1.0 if 10 <= summary_length <= 150 else 0.5
            
            # Complexity adaptation (lower complexity should be easier to summarize)
            complexity_score = 1.0 - (complexity * 0.3)
            
            # Combine scores
            confidence = (
                0.4 * max(compression_score, 0.1) +
                0.3 * length_score +
                0.3 * complexity_score
            )
            
            return min(max(confidence, 0.1), 1.0)
            
        except Exception:
            return 0.5  # Default confidence
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 