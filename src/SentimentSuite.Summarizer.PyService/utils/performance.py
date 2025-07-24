import time
import uuid
import psutil
from typing import Dict, Optional, Any
from collections import defaultdict
import threading

from core.interfaces import IPerformanceMonitor

class PerformanceMonitor(IPerformanceMonitor):
    """Performance monitoring with metrics collection and system tracking"""
    
    def __init__(self):
        self._timers: Dict[str, float] = {}
        self._metrics: Dict[str, list] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def start_timing(self, operation: str) -> str:
        """Start timing an operation and return timer ID"""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            self._timers[timer_id] = time.time()
        
        return timer_id
    
    def end_timing(self, timer_id: str) -> float:
        """End timing and return duration in seconds"""
        end_time = time.time()
        
        with self._lock:
            start_time = self._timers.get(timer_id)
            if start_time is None:
                return 0.0
            
            duration = end_time - start_time
            del self._timers[timer_id]
            
            # Extract operation name from timer_id
            operation = timer_id.rsplit('_', 1)[0]
            self._metrics[f"{operation}_duration"].append(duration)
            
            return duration
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a metric with optional labels"""
        with self._lock:
            # Create metric key with labels
            if labels:
                label_str = "_".join(f"{k}={v}" for k, v in sorted(labels.items()))
                metric_key = f"{name}_{label_str}" if label_str else name
            else:
                metric_key = name
            
            self._metrics[metric_key].append(value)
            
            # Also update counters for counting metrics
            if name.endswith('_count') or name.endswith('_total'):
                self._counters[metric_key] += int(value)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics (for current process)
            disk_usage = psutil.disk_usage('/')
            disk_percent = disk_usage.percent
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "memory_percent": memory_percent,
                "memory_used_gb": round(memory_used_gb, 2),
                "memory_total_gb": round(memory_total_gb, 2),
                "disk_percent": disk_percent,
                "process_memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                "process_memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                "process_cpu_percent": process_cpu,
                "uptime_seconds": time.time() - self._start_time
            }
            
        except Exception as e:
            # Return basic metrics if detailed ones fail
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "error": str(e),
                "uptime_seconds": time.time() - self._start_time
            }
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        with self._lock:
            values = self._metrics.get(metric_name, [])
            
            if not values:
                return {"count": 0}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics with summaries"""
        with self._lock:
            result = {
                "system": self.get_system_metrics(),
                "counters": dict(self._counters),
                "metrics": {}
            }
            
            # Add summaries for all metrics
            for metric_name in self._metrics:
                result["metrics"][metric_name] = self.get_metric_summary(metric_name)
            
            return result
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics"""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._timers.clear()
    
    def get_active_timers(self) -> Dict[str, float]:
        """Get currently active timers and their durations"""
        current_time = time.time()
        
        with self._lock:
            return {
                timer_id: current_time - start_time 
                for timer_id, start_time in self._timers.items()
            } 