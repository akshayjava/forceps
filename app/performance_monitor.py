#!/usr/bin/env python3
"""
FORCEPS Performance Monitor and Benchmarking Tool

Comprehensive performance analysis for multi-TB forensic processing:
- Real-time system resource monitoring
- Processing pipeline bottleneck analysis  
- Throughput measurement and optimization recommendations
- Redis queue depth and worker performance tracking
- GPU utilization and memory efficiency monitoring
- Historical performance trending and reporting
"""
import os
import sys
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import yaml
import redis

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from app.distributed_engine import OptimizedRedisClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    load_average: List[float]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class GPUMetrics:
    """GPU resource metrics."""
    timestamp: float
    gpu_id: int
    gpu_utilization: float
    memory_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_draw_w: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics."""
    timestamp: float
    total_images: int
    processed_images: int
    images_per_second: float
    job_queue_size: int
    results_queue_size: int
    active_workers: int
    estimated_completion_time: float
    
    def completion_percentage(self) -> float:
        return (self.processed_images / self.total_images * 100) if self.total_images > 0 else 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkerMetrics:
    """Individual worker performance metrics."""
    worker_id: str
    timestamp: float
    jobs_processed: int
    images_processed: int
    average_job_time: float
    memory_usage_mb: float
    gpu_memory_mb: float
    last_heartbeat: float
    
    def is_active(self, timeout: float = 300) -> bool:
        return (time.time() - self.last_heartbeat) < timeout
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SystemMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self, sample_interval: float = 5.0):
        self.sample_interval = sample_interval
        self.running = False
        self.metrics_history = []
        self.max_history_size = 1000
        
        # Initialize baseline measurements
        self.baseline_disk_io = psutil.disk_io_counters()
        self.baseline_network_io = psutil.net_io_counters()
        self.start_time = time.time()
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (current_disk_io.read_bytes - self.baseline_disk_io.read_bytes) / 1024 / 1024
        disk_write_mb = (current_disk_io.write_bytes - self.baseline_disk_io.write_bytes) / 1024 / 1024
        
        # Network I/O
        current_network_io = psutil.net_io_counters()
        network_sent_mb = (current_network_io.bytes_sent - self.baseline_network_io.bytes_sent) / 1024 / 1024
        network_recv_mb = (current_network_io.bytes_recv - self.baseline_network_io.bytes_recv) / 1024 / 1024
        
        # Load average
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024**3,
            memory_available_gb=memory.available / 1024**3,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_io_sent_mb=network_sent_mb,
            network_io_recv_mb=network_recv_mb,
            load_average=load_avg
        )
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    metrics = self.get_current_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Limit history size
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history = self.metrics_history[-self.max_history_size:]
                    
                    time.sleep(self.sample_interval)
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    time.sleep(self.sample_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=10)
        logger.info("System monitoring stopped")
    
    def get_average_metrics(self, window_minutes: float = 5.0) -> Optional[SystemMetrics]:
        """Get average metrics over a time window."""
        if not self.metrics_history:
            return None
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            memory_percent=sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            memory_used_gb=sum(m.memory_used_gb for m in recent_metrics) / len(recent_metrics),
            memory_available_gb=sum(m.memory_available_gb for m in recent_metrics) / len(recent_metrics),
            disk_io_read_mb=sum(m.disk_io_read_mb for m in recent_metrics) / len(recent_metrics),
            disk_io_write_mb=sum(m.disk_io_write_mb for m in recent_metrics) / len(recent_metrics),
            network_io_sent_mb=sum(m.network_io_sent_mb for m in recent_metrics) / len(recent_metrics),
            network_io_recv_mb=sum(m.network_io_recv_mb for m in recent_metrics) / len(recent_metrics),
            load_average=[
                sum(m.load_average[0] for m in recent_metrics) / len(recent_metrics),
                sum(m.load_average[1] for m in recent_metrics) / len(recent_metrics),
                sum(m.load_average[2] for m in recent_metrics) / len(recent_metrics)
            ]
        )
        
        return avg_metrics


class GPUMonitor:
    """Monitor GPU resources and performance."""
    
    def __init__(self, sample_interval: float = 5.0):
        self.sample_interval = sample_interval
        self.running = False
        self.metrics_history = []
        self.max_history_size = 1000
        self.gpus_available = GPUTIL_AVAILABLE or TORCH_AVAILABLE
    
    def get_current_gpu_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics."""
        metrics = []
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    metrics.append(GPUMetrics(
                        timestamp=time.time(),
                        gpu_id=gpu.id,
                        gpu_utilization=gpu.load * 100,
                        memory_utilization=gpu.memoryUtil * 100,
                        memory_used_mb=gpu.memoryUsed,
                        memory_total_mb=gpu.memoryTotal,
                        temperature_c=gpu.temperature,
                        power_draw_w=0.0  # Not available in GPUtil
                    ))
            except Exception as e:
                logger.debug(f"GPUtil failed: {e}")
        
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                        memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
                        
                        props = torch.cuda.get_device_properties(i)
                        total_memory = props.total_memory / 1024**2
                        
                        metrics.append(GPUMetrics(
                            timestamp=time.time(),
                            gpu_id=i,
                            gpu_utilization=0.0,  # Not available through PyTorch
                            memory_utilization=memory_allocated / total_memory * 100,
                            memory_used_mb=memory_allocated,
                            memory_total_mb=total_memory,
                            temperature_c=0.0,  # Not available through PyTorch
                            power_draw_w=0.0   # Not available through PyTorch
                        ))
            except Exception as e:
                logger.debug(f"PyTorch GPU monitoring failed: {e}")
        
        return metrics
    
    def start_monitoring(self):
        """Start GPU monitoring."""
        if not self.gpus_available:
            logger.info("No GPU monitoring available")
            return
        
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    metrics = self.get_current_gpu_metrics()
                    self.metrics_history.extend(metrics)
                    
                    # Limit history size
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history = self.metrics_history[-self.max_history_size:]
                    
                    time.sleep(self.sample_interval)
                    
                except Exception as e:
                    logger.error(f"Error in GPU monitoring: {e}")
                    time.sleep(self.sample_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=10)


class ProcessingMonitor:
    """Monitor processing pipeline performance."""
    
    def __init__(self, redis_client: OptimizedRedisClient, config: Dict[str, Any]):
        self.redis_client = redis_client
        self.config = config
        self.metrics_history = []
        self.max_history_size = 1000
        self.running = False
        
        self.job_queue = config['redis']['job_queue']
        self.results_queue = config['redis']['results_queue']
    
    def get_current_processing_metrics(self) -> ProcessingMetrics:
        """Get current processing pipeline metrics."""
        # Get queue sizes
        job_queue_size = self.redis_client.get_queue_size(self.job_queue)
        results_queue_size = self.redis_client.get_queue_size(self.results_queue)
        
        # Get progress counters
        total_images = int(self.redis_client.redis_client.get("forceps:stats:total_images") or 0)
        processed_images = int(self.redis_client.redis_client.get("forceps:stats:embeddings_done") or 0)
        
        # Calculate processing rate
        images_per_second = 0.0
        if len(self.metrics_history) >= 2:
            recent = self.metrics_history[-1]
            older = self.metrics_history[-2]
            time_diff = recent.timestamp - older.timestamp
            images_diff = recent.processed_images - older.processed_images
            if time_diff > 0:
                images_per_second = images_diff / time_diff
        
        # Estimate completion time
        remaining_images = total_images - processed_images
        estimated_completion_time = 0.0
        if images_per_second > 0:
            estimated_completion_time = remaining_images / images_per_second / 60  # minutes
        
        # Count active workers (simplified - would need to check worker heartbeats)
        active_workers = 0
        try:
            worker_keys = self.redis_client.redis_client.keys("forceps:worker:*:stats")
            active_workers = len(worker_keys)
        except Exception:
            pass
        
        return ProcessingMetrics(
            timestamp=time.time(),
            total_images=total_images,
            processed_images=processed_images,
            images_per_second=images_per_second,
            job_queue_size=job_queue_size,
            results_queue_size=results_queue_size,
            active_workers=active_workers,
            estimated_completion_time=estimated_completion_time
        )
    
    def start_monitoring(self, sample_interval: float = 10.0):
        """Start processing monitoring."""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    metrics = self.get_current_processing_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Limit history size
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history = self.metrics_history[-self.max_history_size:]
                    
                    time.sleep(sample_interval)
                    
                except Exception as e:
                    logger.error(f"Error in processing monitoring: {e}")
                    time.sleep(sample_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Processing monitoring started")
    
    def stop_monitoring(self):
        """Stop processing monitoring."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=10)


class PerformanceAnalyzer:
    """Analyze performance data and provide optimization recommendations."""
    
    def __init__(self):
        self.bottleneck_thresholds = {
            'cpu_high': 90.0,
            'memory_high': 85.0,
            'disk_io_high': 100.0,  # MB/s
            'queue_backlog_high': 1000,
            'processing_stall': 60.0  # seconds without progress
        }
    
    def analyze_system_bottlenecks(self, 
                                 system_metrics: List[SystemMetrics],
                                 gpu_metrics: List[GPUMetrics],
                                 processing_metrics: List[ProcessingMetrics]) -> Dict[str, Any]:
        """Analyze system performance and identify bottlenecks."""
        
        analysis = {
            'bottlenecks': [],
            'recommendations': [],
            'performance_score': 100,
            'resource_utilization': {}
        }
        
        if not system_metrics:
            return analysis
        
        recent_system = system_metrics[-10:] if len(system_metrics) >= 10 else system_metrics
        recent_processing = processing_metrics[-10:] if len(processing_metrics) >= 10 else processing_metrics
        
        # Analyze CPU usage
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        if avg_cpu > self.bottleneck_thresholds['cpu_high']:
            analysis['bottlenecks'].append('HIGH_CPU_USAGE')
            analysis['recommendations'].append(
                f"CPU usage is high ({avg_cpu:.1f}%). Consider reducing worker processes or batch sizes."
            )
            analysis['performance_score'] -= 20
        
        # Analyze Memory usage
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        if avg_memory > self.bottleneck_thresholds['memory_high']:
            analysis['bottlenecks'].append('HIGH_MEMORY_USAGE')
            analysis['recommendations'].append(
                f"Memory usage is high ({avg_memory:.1f}%). Consider reducing batch sizes or adding more RAM."
            )
            analysis['performance_score'] -= 15
        
        # Analyze Disk I/O
        avg_disk_read = sum(m.disk_io_read_mb for m in recent_system) / len(recent_system)
        avg_disk_write = sum(m.disk_io_write_mb for m in recent_system) / len(recent_system)
        total_disk_io = avg_disk_read + avg_disk_write
        
        if total_disk_io > self.bottleneck_thresholds['disk_io_high']:
            analysis['bottlenecks'].append('HIGH_DISK_IO')
            analysis['recommendations'].append(
                f"Disk I/O is high ({total_disk_io:.1f} MB/s). Consider using faster storage or optimizing file access patterns."
            )
            analysis['performance_score'] -= 10
        
        # Analyze Queue backlogs
        if recent_processing:
            latest_processing = recent_processing[-1]
            if latest_processing.job_queue_size > self.bottleneck_thresholds['queue_backlog_high']:
                analysis['bottlenecks'].append('QUEUE_BACKLOG')
                analysis['recommendations'].append(
                    f"Job queue has {latest_processing.job_queue_size} pending jobs. Consider adding more workers."
                )
                analysis['performance_score'] -= 15
        
        # Analyze GPU utilization
        if gpu_metrics:
            recent_gpu = gpu_metrics[-10:] if len(gpu_metrics) >= 10 else gpu_metrics
            if recent_gpu:
                avg_gpu_util = sum(m.gpu_utilization for m in recent_gpu) / len(recent_gpu)
                avg_gpu_memory = sum(m.memory_utilization for m in recent_gpu) / len(recent_gpu)
                
                if avg_gpu_util < 50.0:
                    analysis['bottlenecks'].append('LOW_GPU_UTILIZATION')
                    analysis['recommendations'].append(
                        f"GPU utilization is low ({avg_gpu_util:.1f}%). Consider increasing batch sizes or optimizing GPU code."
                    )
                    analysis['performance_score'] -= 10
                
                analysis['resource_utilization']['gpu_compute'] = avg_gpu_util
                analysis['resource_utilization']['gpu_memory'] = avg_gpu_memory
        
        # Record resource utilization
        analysis['resource_utilization'].update({
            'cpu': avg_cpu,
            'memory': avg_memory,
            'disk_io': total_disk_io
        })
        
        return analysis
    
    def generate_optimization_report(self, 
                                   analysis: Dict[str, Any],
                                   processing_metrics: List[ProcessingMetrics]) -> str:
        """Generate a comprehensive optimization report."""
        
        report = []
        report.append("FORCEPS Performance Optimization Report")
        report.append("=" * 50)
        report.append(f"Performance Score: {analysis['performance_score']}/100")
        report.append("")
        
        # Current throughput
        if processing_metrics:
            latest = processing_metrics[-1]
            report.append(f"Current Processing Rate: {latest.images_per_second:.1f} images/second")
            report.append(f"Progress: {latest.processed_images:,}/{latest.total_images:,} images "
                         f"({latest.completion_percentage():.1f}%)")
            if latest.estimated_completion_time > 0:
                report.append(f"Estimated Completion: {latest.estimated_completion_time:.1f} minutes")
            report.append("")
        
        # Resource utilization
        if 'resource_utilization' in analysis:
            report.append("Resource Utilization:")
            for resource, value in analysis['resource_utilization'].items():
                report.append(f"  {resource.upper()}: {value:.1f}%")
            report.append("")
        
        # Bottlenecks
        if analysis['bottlenecks']:
            report.append("Identified Bottlenecks:")
            for bottleneck in analysis['bottlenecks']:
                report.append(f"  - {bottleneck}")
            report.append("")
        
        # Recommendations
        if analysis['recommendations']:
            report.append("Optimization Recommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        # Performance projections
        if processing_metrics and len(processing_metrics) > 10:
            recent_rates = [m.images_per_second for m in processing_metrics[-10:]]
            avg_rate = sum(recent_rates) / len(recent_rates)
            
            report.append("Performance Projections:")
            report.append(f"  With current performance: {avg_rate:.1f} images/second")
            
            # Estimate potential improvements
            if avg_rate > 0:
                optimized_rate = avg_rate * 1.5  # Conservative 50% improvement estimate
                report.append(f"  With optimizations: ~{optimized_rate:.1f} images/second (estimated)")
                
                # Time to process various dataset sizes
                dataset_sizes = [100000, 1000000, 10000000]  # 100K, 1M, 10M images
                report.append("")
                report.append("Estimated Processing Times for Different Dataset Sizes:")
                for size in dataset_sizes:
                    current_time = size / avg_rate / 3600  # hours
                    optimized_time = size / optimized_rate / 3600  # hours
                    report.append(f"  {size:,} images: {current_time:.1f}h current, {optimized_time:.1f}h optimized")
        
        return "\n".join(report)


class PerformanceMonitor:
    """Main performance monitoring coordinator."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Redis client
        redis_config = self.config['redis']
        self.redis_client = OptimizedRedisClient(
            host=redis_config['host'],
            port=redis_config['port']
        )
        
        # Initialize monitors
        self.system_monitor = SystemMonitor()
        self.gpu_monitor = GPUMonitor()
        self.processing_monitor = ProcessingMonitor(self.redis_client, self.config)
        self.analyzer = PerformanceAnalyzer()
        
        self.running = False
        self.report_interval = 60.0  # Generate reports every minute
    
    def start_monitoring(self):
        """Start all monitoring components."""
        logger.info("Starting comprehensive performance monitoring...")
        
        self.system_monitor.start_monitoring()
        self.gpu_monitor.start_monitoring()
        self.processing_monitor.start_monitoring()
        
        self.running = True
        
        # Start reporting thread
        def report_loop():
            while self.running:
                try:
                    time.sleep(self.report_interval)
                    self.generate_and_log_report()
                except Exception as e:
                    logger.error(f"Error in reporting loop: {e}")
        
        self.report_thread = threading.Thread(target=report_loop, daemon=True)
        self.report_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        logger.info("Stopping performance monitoring...")
        self.running = False
        
        self.system_monitor.stop_monitoring()
        self.gpu_monitor.stop_monitoring()
        self.processing_monitor.stop_monitoring()
        
        if hasattr(self, 'report_thread'):
            self.report_thread.join(timeout=10)
        
        logger.info("Performance monitoring stopped")
    
    def generate_and_log_report(self):
        """Generate and log performance report."""
        try:
            analysis = self.analyzer.analyze_system_bottlenecks(
                self.system_monitor.metrics_history,
                self.gpu_monitor.metrics_history,
                self.processing_monitor.metrics_history
            )
            
            report = self.analyzer.generate_optimization_report(
                analysis,
                self.processing_monitor.metrics_history
            )
            
            logger.info("\n" + report)
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
    
    def save_metrics_to_file(self, output_path: str):
        """Save all metrics to a file for analysis."""
        metrics_data = {
            'timestamp': time.time(),
            'system_metrics': [m.to_dict() for m in self.system_monitor.metrics_history],
            'gpu_metrics': [m.to_dict() for m in self.gpu_monitor.metrics_history],
            'processing_metrics': [m.to_dict() for m in self.processing_monitor.metrics_history]
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")


def main():
    """Main entry point for performance monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FORCEPS Performance Monitor")
    parser.add_argument("--config", type=str, default="app/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--duration", type=int, default=0,
                       help="Monitoring duration in seconds (0 = infinite)")
    parser.add_argument("--output", type=str,
                       help="Output file for metrics data")
    parser.add_argument("--report_interval", type=int, default=60,
                       help="Report generation interval in seconds")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.config)
    monitor.report_interval = args.report_interval
    
    try:
        monitor.start_monitoring()
        
        if args.duration > 0:
            logger.info(f"Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            logger.info("Monitoring indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        monitor.stop_monitoring()
        
        if args.output:
            monitor.save_metrics_to_file(args.output)


if __name__ == "__main__":
    main()
