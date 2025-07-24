import asyncio
import hashlib
import json
import time
from typing import Optional, Dict, Any
from cachetools import TTLCache
from dataclasses import asdict

from core.interfaces import ICacheService
from core.config import settings

class MemoryCacheService(ICacheService):
    """High-performance in-memory cache with TTL and LRU eviction"""
    
    def __init__(self):
        self._cache = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl
        )
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[str]:
        """Get cached value with async lock"""
        async with self._lock:
            try:
                value = self._cache.get(key)
                if value is not None:
                    self._stats["hits"] += 1
                    return value
                else:
                    self._stats["misses"] += 1
                    return None
            except Exception:
                self._stats["misses"] += 1
                return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set cached value with optional custom TTL"""
        async with self._lock:
            try:
                if ttl is not None:
                    # For custom TTL, we need to handle it manually
                    expire_time = time.time() + ttl
                    wrapped_value = {
                        "value": value,
                        "expires_at": expire_time
                    }
                    self._cache[key] = json.dumps(wrapped_value)
                else:
                    self._cache[key] = value
                
                self._stats["sets"] += 1
            except Exception as e:
                # Cache is full or other error - this is acceptable
                self._stats["evictions"] += 1
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        async with self._lock:
            return key in self._cache
    
    def generate_key(self, text: str, params: Dict[str, Any]) -> str:
        """Generate consistent cache key from text and parameters"""
        # Create a stable representation of parameters
        sorted_params = dict(sorted(params.items()))
        
        # Create content hash
        content = {
            "text": text.strip(),
            "params": sorted_params
        }
        
        content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA256 hash for consistent, collision-resistant keys
        hash_obj = hashlib.sha256(content_str.encode('utf-8'))
        return f"summary_{hash_obj.hexdigest()[:16]}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self._cache.maxsize,
                "hit_rate": hit_rate,
                "total_hits": self._stats["hits"],
                "total_misses": self._stats["misses"],
                "total_sets": self._stats["sets"],
                "total_evictions": self._stats["evictions"]
            }
    
    async def clear(self) -> None:
        """Clear all cached entries"""
        async with self._lock:
            self._cache.clear()
    
    async def _cleanup_expired(self) -> None:
        """Clean up manually managed TTL entries"""
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, value in self._cache.items():
                try:
                    if isinstance(value, str) and value.startswith('{"value":'):
                        wrapped = json.loads(value)
                        if "expires_at" in wrapped and wrapped["expires_at"] < current_time:
                            expired_keys.append(key)
                except (json.JSONDecodeError, KeyError):
                    continue
            
            for key in expired_keys:
                del self._cache[key] 