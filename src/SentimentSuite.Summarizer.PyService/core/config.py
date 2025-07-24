"""
Configuration management for SentimentSuite Summarizer service.

This module provides environment-based configuration using Pydantic Settings,
enabling flexible deployment across different environments (dev, staging, prod)
while maintaining type safety and validation.

Key Features:
- Environment variable mapping with defaults
- Type validation and coercion
- Hierarchical configuration (env file -> env vars -> defaults)
- Runtime device detection for optimal model placement
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    """
    Application settings with comprehensive environment variable support.
    
    Configuration is loaded in order of priority:
    1. Environment variables (highest priority)
    2. .env file values  
    3. Default values (lowest priority)
    
    All settings are validated at startup to catch configuration errors early.
    """
    
    # ==================== Server Configuration ====================
    host: str = Field(default="0.0.0.0", env="HOST", description="Server bind address")
    port: int = Field(default=8000, env="PORT", description="Server port number", ge=1, le=65535)
    workers: int = Field(default=1, env="WORKERS", description="Number of worker processes", ge=1, le=16)
    
    # ==================== Model Configuration ====================  
    model_name: str = Field(
        default="facebook/bart-large-cnn", 
        env="MODEL_NAME",
        description="HuggingFace model identifier for summarization"
    )
    model_cache_dir: Optional[str] = Field(
        default=None, 
        env="MODEL_CACHE_DIR",
        description="Directory for caching downloaded models (None = default)"
    )
    device: str = Field(
        default="auto", 
        env="DEVICE",
        description="Device for model inference: 'auto', 'cpu', or 'cuda'"
    )
    model_precision: str = Field(
        default="float32", 
        env="MODEL_PRECISION",
        description="Model precision: 'float32' or 'float16' (for memory optimization)"
    )
    
    # ==================== Processing Configuration ====================
    max_workers: int = Field(
        default=3, 
        env="MAX_WORKERS",
        description="Maximum parallel workers for chunk processing",
        ge=1, le=32
    )
    max_chunk_size: int = Field(
        default=512, 
        env="MAX_CHUNK_SIZE",
        description="Maximum tokens per chunk for processing",
        ge=128, le=2048
    )
    chunk_overlap: int = Field(
        default=50, 
        env="CHUNK_OVERLAP",
        description="Token overlap between chunks for context preservation",
        ge=0, le=200
    )
    batch_size: int = Field(
        default=4, 
        env="BATCH_SIZE",
        description="Batch size for parallel chunk processing",
        ge=1, le=16
    )
    
    # ==================== Cache Configuration ====================
    cache_enabled: bool = Field(
        default=True, 
        env="CACHE_ENABLED",
        description="Enable intelligent caching for performance"
    )
    cache_ttl: int = Field(
        default=3600, 
        env="CACHE_TTL",
        description="Cache time-to-live in seconds (1 hour default)",
        ge=60, le=86400
    )
    cache_max_size: int = Field(
        default=1000, 
        env="CACHE_MAX_SIZE",
        description="Maximum number of cached entries",
        ge=10, le=10000
    )
    
    # ==================== Performance Configuration ====================
    enable_metrics: bool = Field(
        default=True, 
        env="ENABLE_METRICS",
        description="Enable performance metrics collection"
    )
    metrics_port: int = Field(
        default=8001, 
        env="METRICS_PORT",
        description="Port for metrics endpoint",
        ge=1, le=65535
    )
    log_level: str = Field(
        default="INFO", 
        env="LOG_LEVEL",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    
    # ==================== Optimization Configuration ====================
    enable_quantization: bool = Field(
        default=False, 
        env="ENABLE_QUANTIZATION",
        description="Enable model quantization for memory/speed optimization"
    )
    enable_onnx: bool = Field(
        default=False, 
        env="ENABLE_ONNX",
        description="Enable ONNX optimization (experimental)"
    )
    prefetch_batches: int = Field(
        default=2, 
        env="PREFETCH_BATCHES",
        description="Number of batches to prefetch for pipeline optimization",
        ge=1, le=8
    )
    
    class Config:
        env_file = ".env"  # Load from .env file if present
        case_sensitive = False  # Allow case-insensitive env var names
        # Fix Pydantic v2 namespace warnings for model_* fields
        protected_namespaces = ('settings_',)  # Only protect 'settings_' namespace

# ==================== Global Configuration Instance ====================
# Singleton pattern for application-wide configuration access
settings = Settings()

# ==================== Utility Functions ====================

def get_model_device() -> str:
    """
    Intelligently determine the optimal device for model inference.
    
    Auto-detection logic:
    1. If device="auto", detect CUDA availability
    2. If CUDA available and has sufficient memory, use GPU  
    3. Fall back to CPU for compatibility
    4. Respect explicit device settings
    
    Returns:
        Device string: 'cuda', 'cpu', or specific device like 'cuda:0'
    """
    if settings.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                # Additional check for CUDA memory availability
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Get memory info for default device
                    memory_free, memory_total = torch.cuda.mem_get_info()
                    memory_free_gb = memory_free / (1024**3)
                    
                    # Require at least 2GB free memory for model loading
                    if memory_free_gb > 2.0:
                        return "cuda"
                        
            return "cpu"  # Fallback to CPU
        except ImportError:
            # PyTorch not available, use CPU
            return "cpu"
    return settings.device

def get_model_dtype():
    """
    Get the appropriate PyTorch data type for model optimization.
    
    Float16 can significantly reduce memory usage and increase speed
    on compatible hardware, but may affect quality slightly.
    
    Returns:
        PyTorch dtype or None for default
    """
    if settings.model_precision == "float16":
        try:
            import torch
            # Verify device supports float16 efficiently
            device = get_model_device()
            if device.startswith("cuda"):
                # GPU supports float16 well
                return torch.float16
            else:
                # CPU float16 can be slower, skip optimization
                return None
        except ImportError:
            pass
    return None  # Use default float32

def validate_configuration() -> bool:
    """
    Validate configuration for common issues and compatibility.
    
    Performs runtime validation of configuration combinations
    that can't be caught by Pydantic field validation.
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration has incompatible settings
    """
    # Validate model and device compatibility
    device = get_model_device()
    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise ValueError("CUDA device specified but not available")
        except ImportError:
            raise ValueError("CUDA device specified but PyTorch not available")
    
    # Validate worker limits for system resources
    import os
    cpu_count = os.cpu_count() or 4
    if settings.max_workers > cpu_count * 2:
        import warnings
        warnings.warn(
            f"max_workers ({settings.max_workers}) exceeds 2x CPU count ({cpu_count}). "
            "This may cause performance degradation."
        )
    
    # Validate cache settings
    if settings.cache_enabled and settings.cache_max_size < 10:
        raise ValueError("cache_max_size must be at least 10 when caching is enabled")
    
    return True 