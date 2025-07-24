"""
Core interfaces for the SentimentSuite Summarizer service.

This module defines the abstract base classes that form the foundation of our clean architecture.
Following SOLID principles, these interfaces enable dependency inversion and make the system
highly testable and extensible.

Architecture Overview:
- ITextChunker: Strategy pattern for different text segmentation approaches
- ISummarizer: Core abstraction for ML model implementations  
- ICacheService: Repository pattern for caching layer abstraction
- IPerformanceMonitor: Observer pattern for metrics and monitoring
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ChunkStrategy(Enum):
    """
    Text chunking strategies available in the system.
    
    SEMANTIC: Uses sentence embeddings and clustering for intelligent segmentation
    ADAPTIVE: Dynamically adjusts chunk size based on content complexity  
    FIXED_SIZE: Traditional fixed-size chunking with overlap
    """
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"
    FIXED_SIZE = "fixed_size"

class SummaryQuality(Enum):
    """
    Quality vs Speed trade-off levels for summarization.
    
    FAST: Optimized for speed with minimal quality loss
    BALANCED: Good balance between quality and performance (default)
    QUALITY: Maximum quality with longer processing time
    """
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"

@dataclass
class ChunkMetadata:
    """
    Rich metadata for text chunks enabling intelligent processing.
    
    This metadata drives adaptive summarization parameters and quality control.
    """
    id: int  # Unique identifier for ordering and tracking
    start_pos: int  # Character position in original text
    end_pos: int  # End character position in original text
    token_count: int  # Approximate token count for model planning
    sentence_count: int  # Number of sentences for complexity estimation
    complexity_score: float  # 0.0-1.0 complexity score for adaptive processing
    priority: int = 1  # Processing priority (1=normal, higher=priority)

@dataclass
class SummaryResult:
    """
    Container for summarization results with quality metrics.
    
    Includes confidence scoring and performance metrics for monitoring and optimization.
    """
    content: str  # The generated summary text
    confidence_score: float  # 0.0-1.0 quality confidence estimate
    processing_time: float  # Processing duration in seconds
    chunk_id: Optional[int] = None  # Reference to source chunk
    metadata: Dict[str, Any] = None  # Additional processing metadata

@dataclass
class TextChunk:
    """
    Container combining text content with rich metadata.
    
    This is the fundamental unit of processing in our chunking pipeline.
    """
    content: str  # The actual text content
    metadata: ChunkMetadata  # Rich metadata for intelligent processing

class ITextChunker(ABC):
    """
    Abstract interface for text chunking strategies (Strategy Pattern).
    
    Enables different approaches to text segmentation while maintaining
    consistent interface for the orchestration layer.
    """
    
    @abstractmethod
    async def chunk_text(self, text: str, max_tokens: int = 512) -> List[TextChunk]:
        """
        Split text into optimized chunks with metadata.
        
        Args:
            text: Input text to be chunked
            max_tokens: Maximum tokens per chunk (advisory)
            
        Returns:
            List of TextChunk objects with rich metadata
            
        Raises:
            ValueError: If text is empty or invalid
        """
        pass
    
    @abstractmethod
    def estimate_optimal_chunk_size(self, text: str) -> int:
        """
        Analyze content to determine optimal chunk size.
        
        Different content types (technical, narrative, etc.) benefit from
        different chunk sizes. This method analyzes complexity and structure.
        
        Args:
            text: Text to analyze
            
        Returns:
            Recommended chunk size in tokens
        """
        pass

class ISummarizer(ABC):
    """
    Abstract interface for summarization implementations (Strategy Pattern).
    
    Abstracts the ML model layer enabling different models, optimizations,
    and processing strategies while maintaining consistent interface.
    """
    
    @abstractmethod
    async def summarize_chunk(self, chunk: TextChunk, quality: SummaryQuality = SummaryQuality.BALANCED) -> SummaryResult:
        """
        Summarize a single text chunk with quality control.
        
        Args:
            chunk: TextChunk with content and metadata
            quality: Quality vs speed preference
            
        Returns:
            SummaryResult with content and metrics
            
        Raises:
            RuntimeError: If summarization fails
        """
        pass
    
    @abstractmethod
    async def batch_summarize(self, chunks: List[TextChunk], quality: SummaryQuality = SummaryQuality.BALANCED) -> List[SummaryResult]:
        """
        Process multiple chunks efficiently with batching.
        
        Enables parallel processing and model optimization through batching.
        Should maintain chunk order in results.
        
        Args:
            chunks: List of TextChunks to process
            quality: Quality preference for all chunks
            
        Returns:
            List of SummaryResults in original chunk order
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the summarizer is initialized and ready for processing.
        
        Returns:
            True if ready, False if still initializing
        """
        pass

class ICacheService(ABC):
    """
    Abstract interface for caching implementations (Repository Pattern).
    
    Provides abstraction over different caching backends (memory, Redis, etc.)
    with consistent interface for the business logic layer.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """
        Retrieve cached value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """
        Store value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache (must be serializable)
            ttl: Time-to-live in seconds (None uses default)
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists and is not expired
        """
        pass
    
    @abstractmethod
    def generate_key(self, text: str, params: Dict[str, Any]) -> str:
        """
        Generate consistent cache key from content and parameters.
        
        Must produce identical keys for identical inputs to ensure cache hits.
        Should be collision-resistant and reasonably short.
        
        Args:
            text: Input text content
            params: Processing parameters
            
        Returns:
            Consistent, collision-resistant cache key
        """
        pass

class IPerformanceMonitor(ABC):
    """
    Abstract interface for performance monitoring (Observer Pattern).
    
    Enables comprehensive monitoring and metrics collection without
    coupling business logic to specific monitoring implementations.
    """
    
    @abstractmethod
    def start_timing(self, operation: str) -> str:
        """
        Begin timing an operation.
        
        Args:
            operation: Operation name for metrics grouping
            
        Returns:
            Timer ID for ending the timing
        """
        pass
    
    @abstractmethod
    def end_timing(self, timer_id: str) -> float:
        """
        End timing and record duration.
        
        Args:
            timer_id: Timer ID from start_timing()
            
        Returns:
            Duration in seconds
        """
        pass
    
    @abstractmethod
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Record a metric value with optional labels.
        
        Supports both counter and gauge metrics with flexible labeling
        for detailed monitoring and alerting.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for metric segmentation
        """
        pass
    
    @abstractmethod
    def get_system_metrics(self) -> Dict[str, float]:
        """
        Get current system performance metrics.
        
        Returns:
            Dictionary of system metrics (CPU, memory, etc.)
        """
        pass 