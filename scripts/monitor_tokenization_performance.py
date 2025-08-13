#!/usr/bin/env python3
"""
Performance monitoring script for ETHOS tokenization.
Tracks CPU, memory, and I/O usage during tokenization.
"""

import psutil
import time
import os
import json
from pathlib import Path
from datetime import datetime
import threading
import argparse


class PerformanceMonitor:
    def __init__(self, output_file="tokenization_performance.json"):
        self.output_file = output_file
        self.monitoring = False
        self.metrics = []
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring in a separate thread."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Performance monitoring started. Metrics will be saved to {self.output_file}")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        self._save_metrics()
        print("Performance monitoring stopped.")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                # Get process metrics for Python processes
                python_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                    try:
                        if 'python' in proc.info['name'].lower():
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'cpu_percent': proc.info['cpu_percent'],
                                'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Calculate total Python process usage
                total_python_cpu = sum(p['cpu_percent'] for p in python_processes)
                total_python_memory = sum(p['memory_mb'] for p in python_processes)
                
                metric = {
                    'timestamp': time.time(),
                    'elapsed_seconds': time.time() - self.start_time,
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                        'memory_used_gb': memory.used / 1024 / 1024 / 1024,
                        'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                        'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                    },
                    'python_processes': {
                        'count': len(python_processes),
                        'total_cpu_percent': total_python_cpu,
                        'total_memory_mb': total_python_memory,
                        'processes': python_processes
                    }
                }
                
                self.metrics.append(metric)
                
                # Print current status
                elapsed = time.time() - self.start_time
                print(f"\r[{elapsed:.0f}s] CPU: {cpu_percent:.1f}% | "
                      f"Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB) | "
                      f"Python: {len(python_processes)} procs, {total_python_cpu:.1f}% CPU, {total_python_memory:.1f}MB RAM", 
                      end='', flush=True)
                
            except Exception as e:
                print(f"\nError in monitoring: {e}")
                time.sleep(1)
                
    def _save_metrics(self):
        """Save metrics to JSON file."""
        if not self.metrics:
            return
            
        summary = {
            'monitoring_start': self.start_time,
            'monitoring_end': time.time(),
            'total_duration_seconds': time.time() - self.start_time,
            'metrics_count': len(self.metrics),
            'metrics': self.metrics,
            'summary': self._calculate_summary()
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def _calculate_summary(self):
        """Calculate summary statistics."""
        if not self.metrics:
            return {}
            
        cpu_values = [m['system']['cpu_percent'] for m in self.metrics]
        memory_values = [m['system']['memory_percent'] for m in self.metrics]
        python_cpu_values = [m['python_processes']['total_cpu_percent'] for m in self.metrics]
        python_memory_values = [m['python_processes']['total_memory_mb'] for m in self.metrics]
        
        return {
            'cpu': {
                'mean': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'mean': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'python_cpu': {
                'mean': sum(python_cpu_values) / len(python_cpu_values),
                'max': max(python_cpu_values),
                'min': min(python_cpu_values)
            },
            'python_memory': {
                'mean': sum(python_memory_values) / len(python_memory_values),
                'max': max(python_memory_values),
                'min': min(python_memory_values)
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Monitor ETHOS tokenization performance")
    parser.add_argument("--output", default="tokenization_performance.json", 
                       help="Output file for performance metrics")
    parser.add_argument("--duration", type=int, default=0,
                       help="Duration to monitor in seconds (0 = until interrupted)")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.output)
    
    try:
        print("Starting performance monitoring...")
        print("Press Ctrl+C to stop monitoring")
        
        monitor.start_monitoring()
        
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            # Wait indefinitely until interrupted
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping performance monitoring...")
    finally:
        monitor.stop_monitoring()
        
        # Print summary
        if monitor.metrics:
            summary = monitor._calculate_summary()
            print(f"\nPerformance Summary:")
            print(f"  Duration: {summary.get('total_duration_seconds', 0):.1f} seconds")
            print(f"  CPU Usage: {summary['cpu']['mean']:.1f}% avg, {summary['cpu']['max']:.1f}% max")
            print(f"  Memory Usage: {summary['memory']['mean']:.1f}% avg, {summary['memory']['max']:.1f}% max")
            print(f"  Python CPU: {summary['python_cpu']['mean']:.1f}% avg, {summary['python_cpu']['max']:.1f}% max")
            print(f"  Python Memory: {summary['python_memory']['mean']:.1f}MB avg, {summary['python_memory']['max']:.1f}MB max")
            print(f"  Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()


