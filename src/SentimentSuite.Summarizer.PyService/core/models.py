from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum
from .interfaces import SummaryQuality, ChunkStrategy

class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000, description="Text to summarize")
    quality: SummaryQuality = Field(default=SummaryQuality.BALANCED, description="Summary quality level")
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.SEMANTIC, description="Chunking strategy")
    max_length: Optional[int] = Field(default=None, ge=20, le=500, description="Maximum summary length")
    min_length: Optional[int] = Field(default=None, ge=10, le=200, description="Minimum summary length")
    use_cache: bool = Field(default=True, description="Whether to use caching")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()
    
    @validator('max_length', 'min_length')
    def validate_lengths(cls, v, values):
        if v is not None and 'max_length' in values and values['max_length'] is not None:
            if v >= values['max_length']:
                raise ValueError("min_length must be less than max_length")
        return v

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="Generated summary")
    processing_time: float = Field(..., description="Processing time in seconds")
    chunk_count: int = Field(..., description="Number of chunks processed")
    cache_hit: bool = Field(..., description="Whether result came from cache")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    cache_status: str = Field(..., description="Cache service status")
    system_metrics: Dict[str, float] = Field(..., description="System performance metrics")
    uptime: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        protected_namespaces = ()  # Disable namespace protection to avoid warnings

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")

class MetricsResponse(BaseModel):
    total_requests: int = Field(..., description="Total requests processed")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    average_processing_time: float = Field(..., description="Average processing time")
    active_connections: int = Field(..., description="Current active connections")
    memory_usage: float = Field(..., description="Memory usage percentage")
    cpu_usage: float = Field(..., description="CPU usage percentage") 