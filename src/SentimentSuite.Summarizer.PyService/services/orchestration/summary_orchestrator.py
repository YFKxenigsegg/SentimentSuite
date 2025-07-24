"""
Main orchestration service for the summarization pipeline.

This module implements the business logic orchestration layer, coordinating between
chunking, caching, summarization, and monitoring services. It applies enterprise
patterns like Circuit Breaker, Retry with Exponential Backoff, and comprehensive
error handling to ensure reliability at scale.

Key Patterns Implemented:
- Circuit Breaker: Prevents cascade failures during service degradation
- Retry with Exponential Backoff: Handles transient failures gracefully  
- Cache-Aside Pattern: Optimizes performance with intelligent caching
- Hierarchical Summarization: Combines chunk results intelligently
- Performance Monitoring: Comprehensive metrics collection

Architecture:
The orchestrator acts as the main coordinator, implementing the facade pattern
to provide a simple interface while managing complex interactions between
multiple specialized services.
"""

import asyncio
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from core.interfaces import (
    ITextChunker, ISummarizer, ICacheService, IPerformanceMonitor,
    TextChunk, SummaryResult, SummaryQuality, ChunkStrategy
)
from core.models import SummaryRequest, SummaryResponse
from core.config import settings

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascade failures.
    
    The circuit breaker pattern prevents a service from repeatedly trying
    to execute an operation that's likely to fail, allowing it to fail fast
    and recover gracefully when the underlying issue is resolved.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failure threshold exceeded, calls fail immediately  
    - HALF_OPEN: Recovery testing, limited calls allowed
    
    This prevents resource exhaustion and allows faster recovery.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker with configurable thresholds.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is OPEN or function fails
        """
        # Check if circuit is open and should remain closed
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                # Transition to half-open for recovery testing
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN for recovery testing")
            else:
                # Circuit still open, fail fast
                raise Exception("Circuit breaker is OPEN - failing fast to prevent cascade failure")
        
        try:
            # Execute the protected function
            result = await func(*args, **kwargs)
            
            # Success - reset circuit breaker if in recovery mode
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker recovery successful - returning to CLOSED state")
                
            return result
            
        except Exception as e:
            # Function failed - increment failure count
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            # Re-raise the exception
            raise e

class SummaryOrchestrator:
    """
    Main orchestrator for the summarization pipeline.
    
    This class implements the business logic layer, coordinating between multiple
    specialized services to provide a complete summarization solution. It handles:
    
    - Request validation and preprocessing
    - Intelligent caching with cache-aside pattern
    - Text chunking with semantic analysis
    - Parallel summarization with retry logic
    - Hierarchical summary assembly
    - Performance monitoring and metrics
    - Circuit breaker protection for resilience
    
    The orchestrator ensures reliability, performance, and observability while
    maintaining clean separation of concerns.
    """
    
    def __init__(
        self,
        chunker: ITextChunker,
        summarizer: ISummarizer,
        cache_service: ICacheService,
        performance_monitor: IPerformanceMonitor
    ):
        """
        Initialize orchestrator with injected dependencies.
        
        Uses dependency injection for loose coupling and testability.
        
        Args:
            chunker: Text chunking service implementation
            summarizer: ML summarization service implementation  
            cache_service: Caching service implementation
            performance_monitor: Metrics collection service
        """
        self.chunker = chunker
        self.summarizer = summarizer
        self.cache_service = cache_service
        self.performance_monitor = performance_monitor
        
        # Initialize circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker()
        
        # Configure retry behavior for transient failures
        self.max_retries = 3
        self.retry_delay = 1.0  # Base delay for exponential backoff
        
    async def process_summary_request(self, request: SummaryRequest) -> SummaryResponse:
        """
        Process a complete summarization request with full orchestration.
        
        This is the main entry point that coordinates the entire pipeline:
        1. Performance timing and monitoring setup
        2. Cache lookup for performance optimization
        3. Text processing with circuit breaker protection
        4. Result caching for future requests
        5. Metrics recording for observability
        
        Args:
            request: Validated summarization request
            
        Returns:
            Complete summarization response with metadata
            
        Raises:
            HTTPException: For various processing failures
        """
        start_time = time.time()
        # Start performance timing for end-to-end monitoring
        timer_id = self.performance_monitor.start_timing("total_request")
        
        try:
            # ========== PHASE 1: Cache Lookup ==========
            cache_hit = False
            if request.use_cache and settings.cache_enabled:
                logger.debug("Attempting cache lookup for request")
                cached_result = await self._try_get_from_cache(request)
                
                if cached_result:
                    # Cache hit - return immediately for optimal performance
                    cache_hit = True
                    processing_time = time.time() - start_time
                    
                    logger.info(f"Cache hit - returning cached result in {processing_time:.3f}s")
                    
                    return SummaryResponse(
                        summary=cached_result["summary"],
                        processing_time=processing_time,
                        chunk_count=cached_result.get("chunk_count", 1),
                        cache_hit=True,
                        quality_score=cached_result.get("quality_score", 0.8),
                        metadata={
                            "cache_hit": True,
                            "cached_at": cached_result.get("cached_at"),
                            "source": "cache"
                        }
                    )
            
            # ========== PHASE 2: Processing Pipeline ==========
            logger.info(f"Processing new request - text length: {len(request.text)} chars")
            
            # Execute processing pipeline with circuit breaker protection
            result = await self.circuit_breaker.call(
                self._process_with_chunking_and_summarization,
                request
            )
            
            processing_time = self.performance_monitor.end_timing(timer_id)
            
            # ========== PHASE 3: Cache Storage ==========
            if request.use_cache and settings.cache_enabled:
                await self._cache_result(request, result, processing_time)
            
            # ========== PHASE 4: Metrics Recording ==========
            self.performance_monitor.record_metric(
                "request_processing_time", 
                processing_time,
                {"cache_hit": str(cache_hit), "quality": request.quality.value}
            )
            
            # ========== PHASE 5: Response Assembly ==========
            logger.info(f"Request completed successfully in {processing_time:.3f}s")
            
            return SummaryResponse(
                summary=result["final_summary"],
                processing_time=processing_time,
                chunk_count=result["chunk_count"],
                cache_hit=False,
                quality_score=result["average_confidence"],
                metadata={
                    "chunk_strategy": request.chunk_strategy.value,
                    "quality_level": request.quality.value,
                    "chunks_processed": result["chunk_count"],
                    "failed_chunks": result.get("failed_chunks", 0),
                    "source": "processed"
                }
            )
            
        except Exception as e:
            # Comprehensive error handling with metrics
            processing_time = self.performance_monitor.end_timing(timer_id)
            logger.error(f"Summary request failed after {processing_time:.3f}s: {e}")
            
            # Record error metrics for monitoring
            self.performance_monitor.record_metric(
                "request_errors", 1,
                {"error_type": type(e).__name__}
            )
            
            # Re-raise for proper HTTP error handling
            raise e
    
    async def _process_with_chunking_and_summarization(self, request: SummaryRequest) -> Dict[str, Any]:
        """
        Core processing pipeline: chunking -> summarization -> assembly.
        
        This method implements the main processing logic:
        1. Intelligent text chunking based on semantic analysis
        2. Parallel chunk summarization with retry logic
        3. Hierarchical summary assembly from chunk results
        4. Quality metrics calculation
        
        Args:
            request: Summarization request with processing parameters
            
        Returns:
            Dictionary with final summary and processing metadata
        """
        # ========== STEP 1: Intelligent Text Chunking ==========
        chunk_timer = self.performance_monitor.start_timing("chunking")
        
        # Determine optimal chunk size based on content analysis
        optimal_chunk_size = self.chunker.estimate_optimal_chunk_size(request.text)
        
        # Perform semantic chunking
        chunks = await self.chunker.chunk_text(request.text, optimal_chunk_size)
        
        self.performance_monitor.end_timing(chunk_timer)
        
        logger.info(f"Created {len(chunks)} chunks using {request.chunk_strategy.value} strategy")
        
        # ========== STEP 2: Parallel Chunk Summarization ==========
        summary_results = await self._summarize_chunks_with_retry(chunks, request.quality)
        
        # ========== STEP 3: Hierarchical Summary Assembly ==========
        final_summary = await self._create_hierarchical_summary(summary_results, request.quality)
        
        # ========== STEP 4: Quality Metrics Calculation ==========
        valid_results = [r for r in summary_results if not r.content.startswith("[Error")]
        failed_chunks = len(summary_results) - len(valid_results)
        average_confidence = sum(r.confidence_score for r in valid_results) / len(valid_results) if valid_results else 0.0
        
        return {
            "final_summary": final_summary,
            "chunk_count": len(chunks),
            "failed_chunks": failed_chunks,
            "average_confidence": average_confidence,
            "summary_results": summary_results
        }
    
    async def _summarize_chunks_with_retry(
        self, 
        chunks: List[TextChunk], 
        quality: SummaryQuality
    ) -> List[SummaryResult]:
        """
        Summarize chunks with comprehensive retry logic and error handling.
        
        Implements a robust processing strategy:
        1. Attempt batch processing for efficiency
        2. Fall back to individual processing on batch failure
        3. Retry failed chunks with exponential backoff
        4. Create error placeholders for permanently failed chunks
        
        Args:
            chunks: List of text chunks to summarize
            quality: Quality level for summarization
            
        Returns:
            List of summary results (including error placeholders)
        """
        # ========== ATTEMPT 1: Batch Processing ==========
        summary_timer = self.performance_monitor.start_timing("batch_summarization")
        
        try:
            # Try efficient batch processing first
            logger.debug(f"Attempting batch processing of {len(chunks)} chunks")
            results = await self.summarizer.batch_summarize(chunks, quality)
            
        except Exception as e:
            logger.warning(f"Batch summarization failed, falling back to individual processing: {e}")
            
            # ========== FALLBACK: Individual Processing ==========
            results = []
            for chunk in chunks:
                try:
                    result = await self.summarizer.summarize_chunk(chunk, quality)
                    results.append(result)
                except Exception as chunk_error:
                    logger.error(f"Failed to summarize chunk {chunk.metadata.id}: {chunk_error}")
                    
                    # Create error placeholder to maintain chunk order
                    error_result = SummaryResult(
                        content=f"[Error processing chunk {chunk.metadata.id}]",
                        confidence_score=0.0,
                        processing_time=0.0,
                        chunk_id=chunk.metadata.id,
                        metadata={"error": str(chunk_error)}
                    )
                    results.append(error_result)
        
        self.performance_monitor.end_timing(summary_timer)
        
        # ========== RETRY LOGIC: Failed Chunk Recovery ==========
        # Identify chunks that failed and need retry
        failed_chunks = [
            (i, chunk) for i, (chunk, result) in enumerate(zip(chunks, results)) 
            if result.content.startswith("[Error")
        ]
        
        if failed_chunks:
            logger.info(f"Retrying {len(failed_chunks)} failed chunks with exponential backoff")
            
            # Implement exponential backoff retry strategy
            for retry_attempt in range(self.max_retries):
                if not failed_chunks:
                    break  # All failures recovered
                
                # Exponential backoff delay: 1s, 2s, 4s
                delay = self.retry_delay * (2 ** retry_attempt)
                await asyncio.sleep(delay)
                
                retry_results = []
                remaining_failed = []
                
                # Attempt to recover each failed chunk
                for result_idx, chunk in failed_chunks:
                    try:
                        retry_result = await self.summarizer.summarize_chunk(chunk, quality)
                        results[result_idx] = retry_result  # Replace error with success
                        retry_results.append(retry_result)
                        
                    except Exception as e:
                        # Still failing - keep for next retry
                        remaining_failed.append((result_idx, chunk))
                        logger.warning(f"Retry {retry_attempt + 1} failed for chunk {chunk.metadata.id}: {e}")
                
                failed_chunks = remaining_failed
                
                if retry_results:
                    logger.info(f"Successfully recovered {len(retry_results)} chunks on retry {retry_attempt + 1}")
        
        return results
    
    async def _create_hierarchical_summary(
        self, 
        chunk_results: List[SummaryResult], 
        quality: SummaryQuality
    ) -> str:
        """
        Create final summary using hierarchical summarization approach.
        
        Implements intelligent summary assembly:
        1. Filter out failed chunk results
        2. For single successful chunk, return directly
        3. For multiple chunks, combine and optionally create meta-summary
        4. Handle edge cases gracefully
        
        Args:
            chunk_results: Results from chunk summarization
            quality: Quality level for meta-summarization
            
        Returns:
            Final assembled summary text
        """
        # Filter out error results to get clean summaries
        valid_results = [r for r in chunk_results if not r.content.startswith("[Error")]
        
        # ========== EDGE CASE: No Valid Results ==========
        if not valid_results:
            logger.error("No valid chunk summaries available")
            return "Unable to generate summary due to processing errors."
        
        # ========== SIMPLE CASE: Single Chunk ==========
        if len(valid_results) == 1:
            logger.debug("Single chunk result - returning directly")
            return valid_results[0].content
        
        # ========== COMPLEX CASE: Multiple Chunks ==========
        logger.debug(f"Assembling summary from {len(valid_results)} chunk results")
        
        # Combine all chunk summaries
        combined_text = " ".join(result.content for result in valid_results)
        
        # If combined text is still long, create meta-summary
        if len(combined_text.split()) > 200:
            logger.debug("Combined summary long - creating meta-summary")
            
            try:
                # Create virtual chunk for meta-summarization
                # This allows us to reuse the same summarization interface
                meta_chunk = TextChunk(
                    content=combined_text,
                    metadata=type('MockMetadata', (), {
                        'id': -1,  # Special ID for meta-chunk
                        'start_pos': 0,
                        'end_pos': len(combined_text),
                        'token_count': len(combined_text.split()),
                        'sentence_count': len(combined_text.split('.')),
                        'complexity_score': 0.5,  # Medium complexity assumption
                        'priority': 1
                    })()
                )
                
                # Perform meta-summarization
                meta_result = await self.summarizer.summarize_chunk(meta_chunk, quality)
                logger.info("Meta-summary created successfully")
                return meta_result.content
                
            except Exception as e:
                logger.warning(f"Meta-summary failed: {e}, returning concatenated summaries")
                # Graceful degradation - return concatenated summaries
                return combined_text
        
        # Combined text is reasonable length - return as-is
        return combined_text
    
    async def _try_get_from_cache(self, request: SummaryRequest) -> Optional[Dict[str, Any]]:
        """
        Attempt to retrieve cached result for the request.
        
        Implements cache-aside pattern with consistent key generation.
        
        Args:
            request: Summarization request
            
        Returns:
            Cached result dictionary or None if not found
        """
        try:
            # Generate consistent cache key from request parameters
            cache_key = self.cache_service.generate_key(
                request.text,
                {
                    "quality": request.quality.value,
                    "chunk_strategy": request.chunk_strategy.value,
                    "max_length": request.max_length,
                    "min_length": request.min_length
                }
            )
            
            # Attempt cache retrieval
            cached_value = await self.cache_service.get(cache_key)
            if cached_value:
                import json
                return json.loads(cached_value)
                
        except Exception as e:
            # Cache failures should not break the main flow
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(
        self, 
        request: SummaryRequest, 
        result: Dict[str, Any], 
        processing_time: float
    ) -> None:
        """
        Store processing result in cache for future requests.
        
        Args:
            request: Original request for key generation
            result: Processing result to cache
            processing_time: Time taken for processing
        """
        try:
            # Generate same cache key as lookup
            cache_key = self.cache_service.generate_key(
                request.text,
                {
                    "quality": request.quality.value,
                    "chunk_strategy": request.chunk_strategy.value,
                    "max_length": request.max_length,
                    "min_length": request.min_length
                }
            )
            
            # Prepare cache value with metadata
            cache_value = {
                "summary": result["final_summary"],
                "chunk_count": result["chunk_count"],
                "quality_score": result["average_confidence"],
                "cached_at": time.time(),
                "processing_time": processing_time
            }
            
            # Store in cache
            import json
            await self.cache_service.set(cache_key, json.dumps(cache_value))
            logger.debug("Result cached successfully")
            
        except Exception as e:
            # Cache storage failures should not break the main flow
            logger.warning(f"Cache storage failed: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all orchestrated components.
        
        Returns:
            Dictionary with health status of all components
        """
        return {
            "summarizer_ready": self.summarizer.is_ready(),
            "circuit_breaker_state": self.circuit_breaker.state,
            "cache_enabled": settings.cache_enabled,
            "system_metrics": self.performance_monitor.get_system_metrics()
        } 