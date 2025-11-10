#!/usr/bin/env python3
"""
Caching Layer for Research Compass.

Provides multi-level caching for:
- Query results
- Embeddings
- Graph data
- LLM responses
- Document processing results
"""

import json
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from functools import wraps
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Multi-level cache manager with TTL support.
    
    Supports:
    - Memory cache (fast, limited size)
    - Disk cache (persistent, larger capacity)
    - TTL-based expiration
    - Cache invalidation
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_items: int = 1000,
        default_ttl_seconds: int = 3600
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache (default: data/cache)
            max_memory_items: Maximum items in memory cache
            default_ttl_seconds: Default TTL for cached items (1 hour)
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl_seconds

        # Memory cache with LRU eviction: OrderedDict maintains insertion order
        # {key: (value, expiry_timestamp)}
        self._memory_cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_lock = threading.Lock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'evictions': 0
        }
        
        logger.info(f"CacheManager initialized: dir={self.cache_dir}, max_items={max_memory_items}")
    
    def _generate_key(self, namespace: str, *args, **kwargs) -> str:
        """
        Generate cache key from namespace and arguments.
        
        Args:
            namespace: Cache namespace (e.g., 'query', 'embedding')
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            SHA256 hash key
        """
        # Create deterministic string from args
        key_parts = [namespace]
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # For complex objects, use repr or json
                try:
                    key_parts.append(json.dumps(arg, sort_keys=True))
                except (TypeError, ValueError):
                    key_parts.append(repr(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                try:
                    key_parts.append(f"{k}={json.dumps(v, sort_keys=True)}")
                except (TypeError, ValueError):
                    key_parts.append(f"{k}={repr(v)}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _is_expired(self, expiry_timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return datetime.now().timestamp() > expiry_timestamp
    
    def _evict_oldest_memory_entry(self):
        """
        Evict least recently used entry from memory cache if at capacity.

        Optimization: Uses OrderedDict for proper LRU eviction policy.
        Prevents unbounded memory growth.
        """
        if len(self._memory_cache) >= self.max_memory_items:
            # OrderedDict: remove first (oldest) item (LRU)
            oldest_key, _ = self._memory_cache.popitem(last=False)
            self.stats['evictions'] += 1
            logger.debug(f"Evicted LRU cache entry: {oldest_key[:16]}...")
    
    def get(
        self,
        namespace: str,
        *args,
        default: Any = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            namespace: Cache namespace
            *args: Key generation arguments
            default: Default value if not found
            **kwargs: Key generation keyword arguments
            
        Returns:
            Cached value or default
        """
        key = self._generate_key(namespace, *args, **kwargs)
        
        with self._cache_lock:
            # Check memory cache first
            if key in self._memory_cache:
                value, expiry = self._memory_cache[key]
                if not self._is_expired(expiry):
                    # Move to end (most recently used) for proper LRU
                    self._memory_cache.move_to_end(key)
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    logger.debug(f"Memory cache hit: {namespace} ({key[:16]}...)")
                    return value
                else:
                    # Expired, remove from memory
                    del self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    value, expiry = data['value'], data['expiry']
                    
                    if not self._is_expired(expiry):
                        # Promote to memory cache
                        with self._cache_lock:
                            self._evict_oldest_memory_entry()
                            self._memory_cache[key] = (value, expiry)
                        
                        self.stats['hits'] += 1
                        self.stats['disk_hits'] += 1
                        logger.debug(f"Disk cache hit: {namespace} ({key[:16]}...)")
                        return value
                    else:
                        # Expired, delete file
                        cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        # Cache miss
        self.stats['misses'] += 1
        logger.debug(f"Cache miss: {namespace} ({key[:16]}...)")
        return default
    
    def set(
        self,
        namespace: str,
        value: Any,
        *args,
        ttl_seconds: Optional[int] = None,
        disk_cache: bool = True,
        **kwargs
    ):
        """
        Set cached value.
        
        Args:
            namespace: Cache namespace
            value: Value to cache
            *args: Key generation arguments
            ttl_seconds: TTL in seconds (default: use default_ttl)
            disk_cache: Whether to also cache to disk
            **kwargs: Key generation keyword arguments
        """
        key = self._generate_key(namespace, *args, **kwargs)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = (datetime.now() + timedelta(seconds=ttl)).timestamp()
        
        # Store in memory cache
        with self._cache_lock:
            self._evict_oldest_memory_entry()
            self._memory_cache[key] = (value, expiry)
        
        # Store in disk cache if requested
        if disk_cache:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'value': value,
                        'expiry': expiry,
                        'namespace': namespace,
                        'created': datetime.now().isoformat()
                    }, f)
                logger.debug(f"Disk cache set: {namespace} ({key[:16]}...)")
            except Exception as e:
                logger.warning(f"Error writing cache file {cache_file}: {e}")
        
        logger.debug(f"Memory cache set: {namespace} ({key[:16]}...), TTL={ttl}s")
    
    def invalidate(self, namespace: str, *args, **kwargs):
        """
        Invalidate specific cache entry.
        
        Args:
            namespace: Cache namespace
            *args: Key generation arguments
            **kwargs: Key generation keyword arguments
        """
        key = self._generate_key(namespace, *args, **kwargs)
        
        # Remove from memory
        with self._cache_lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
        
        # Remove from disk
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        
        logger.debug(f"Invalidated cache: {namespace} ({key[:16]}...)")
    
    def invalidate_namespace(self, namespace: str):
        """
        Invalidate all entries in a namespace.
        
        Args:
            namespace: Cache namespace to clear
        """
        # Clear memory cache entries
        with self._cache_lock:
            keys_to_delete = []
            for key in self._memory_cache.keys():
                # We don't store namespace in memory, so clear all
                # In production, consider storing namespace metadata
                keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._memory_cache[key]
        
        # Clear disk cache entries
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if data.get('namespace') == namespace:
                        cache_file.unlink()
                        count += 1
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")
        
        logger.info(f"Invalidated namespace '{namespace}': {count} disk entries removed")
    
    def clear_all(self):
        """Clear all caches (memory and disk)."""
        with self._cache_lock:
            self._memory_cache.clear()
        
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cleared all caches: {count} disk entries removed")
    
    def clear_expired(self):
        """Remove expired entries from disk cache."""
        now = datetime.now().timestamp()
        count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if self._is_expired(data['expiry']):
                        cache_file.unlink()
                        count += 1
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {count} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': f"{hit_rate:.2f}%",
            'memory_size': len(self._memory_cache),
            'disk_files': len(list(self.cache_dir.glob("*.pkl")))
        }


def cached(
    namespace: str,
    ttl_seconds: Optional[int] = None,
    disk_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
):
    """
    Decorator for caching function results.
    
    Args:
        namespace: Cache namespace
        ttl_seconds: TTL in seconds
        disk_cache: Whether to use disk cache
        cache_manager: CacheManager instance (or use default)
        
    Example:
        @cached('query_results', ttl_seconds=1800)
        def expensive_query(question: str) -> str:
            # ... expensive operation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create cache manager
            cm = cache_manager or getattr(wrapper, '_cache_manager', None)
            if cm is None:
                cm = CacheManager()
                wrapper._cache_manager = cm
            
            # Try to get from cache
            cached_value = cm.get(namespace, *args, **kwargs)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cm.set(
                namespace,
                result,
                *args,
                ttl_seconds=ttl_seconds,
                disk_cache=disk_cache,
                **kwargs
            )
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


if __name__ == '__main__':
    # Example usage
    cache = CacheManager()
    
    # Set values
    cache.set('embeddings', [0.1, 0.2, 0.3], 'document1', ttl_seconds=3600)
    cache.set('query', {'answer': 'Test answer'}, 'What is AI?', ttl_seconds=1800)
    
    # Get values
    emb = cache.get('embeddings', 'document1')
    print(f"Embeddings: {emb}")
    
    query_result = cache.get('query', 'What is AI?')
    print(f"Query result: {query_result}")
    
    # Stats
    print(f"Cache stats: {cache.get_stats()}")
    
    # Test decorator
    @cached('expensive_op', ttl_seconds=60)
    def expensive_operation(x: int) -> int:
        print(f"Computing {x} * 2...")
        return x * 2
    
    print(expensive_operation(5))  # Computes
    print(expensive_operation(5))  # From cache
    print(expensive_operation(10))  # Computes
