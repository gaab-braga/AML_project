# src/orchestration/resource_manager.py
"""
Resource Manager for AML Pipeline Orchestration
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    GPUtil = None

from ..utils import logger
from ..features.metrics import feature_monitor


@dataclass
class ResourceRequirements:
    """Resource requirements for a task"""
    cpu_cores: int = 1
    memory_mb: int = 512
    gpu_required: bool = False
    gpu_memory_mb: int = 0
    disk_space_mb: int = 100
    network_bandwidth_mbps: int = 10
    priority: int = 1  # Higher = more important

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_cores': self.cpu_cores,
            'memory_mb': self.memory_mb,
            'gpu_required': self.gpu_required,
            'gpu_memory_mb': self.gpu_memory_mb,
            'disk_space_mb': self.disk_space_mb,
            'network_bandwidth_mbps': self.network_bandwidth_mbps,
            'priority': self.priority
        }


@dataclass
class ResourceAllocation:
    """Resource allocation for a task"""
    task_id: str
    requirements: ResourceRequirements
    allocated_at: datetime = field(default_factory=datetime.now)
    cpu_cores_allocated: int = 0
    memory_mb_allocated: int = 0
    gpu_allocated: bool = False
    gpu_memory_mb_allocated: int = 0

    def release(self):
        """Release allocated resources"""
        self.cpu_cores_allocated = 0
        self.memory_mb_allocated = 0
        self.gpu_allocated = False
        self.gpu_memory_mb_allocated = 0


@dataclass
class SystemResources:
    """Current system resource status"""
    total_cpu_cores: int
    available_cpu_cores: int
    total_memory_mb: int
    available_memory_mb: int
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_memory_mb: List[int] = field(default_factory=list)
    disk_space_mb: int = 0
    network_bandwidth_mbps: int = 100
    last_updated: datetime = field(default_factory=datetime.now)

    def can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if resources can be allocated"""
        return (
            self.available_cpu_cores >= requirements.cpu_cores and
            self.available_memory_mb >= requirements.memory_mb and
            (not requirements.gpu_required or self.gpu_available) and
            self.disk_space_mb >= requirements.disk_space_mb
        )

    def allocate(self, requirements: ResourceRequirements) -> ResourceAllocation:
        """Allocate resources"""
        if not self.can_allocate(requirements):
            raise ResourceError("Insufficient resources available")

        allocation = ResourceAllocation(
            task_id="",  # Set by caller
            requirements=requirements,
            cpu_cores_allocated=requirements.cpu_cores,
            memory_mb_allocated=requirements.memory_mb,
            gpu_allocated=requirements.gpu_required,
            gpu_memory_mb_allocated=requirements.gpu_memory_mb
        )

        # Update available resources
        self.available_cpu_cores -= requirements.cpu_cores
        self.available_memory_mb -= requirements.memory_mb

        return allocation

    def release_allocation(self, allocation: ResourceAllocation):
        """Release resource allocation"""
        self.available_cpu_cores += allocation.cpu_cores_allocated
        self.available_memory_mb += allocation.memory_mb_allocated
        allocation.release()


class ResourceError(Exception):
    """Exception raised when resource allocation fails"""
    pass


class ResourceManager:
    """
    Resource manager for coordinating computational resources

    Manages CPU, memory, GPU, and other resources across tasks
    """

    def __init__(self,
                 max_cpu_cores: Optional[int] = None,
                 max_memory_mb: Optional[int] = None,
                 enable_gpu: bool = True,
                 monitoring_interval: float = 5.0):
        """
        Initialize resource manager

        Args:
            max_cpu_cores: Maximum CPU cores to use (default: all available)
            max_memory_mb: Maximum memory to use (default: 80% of available)
            enable_gpu: Whether to enable GPU resource management
            monitoring_interval: Resource monitoring interval in seconds
        """
        self.max_cpu_cores = max_cpu_cores
        self.max_memory_mb = max_memory_mb
        self.enable_gpu = enable_gpu
        self.monitoring_interval = monitoring_interval

        self.logger = logger.getChild("resource_manager")

        # Resource tracking
        self._current_resources = self._detect_system_resources()
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._resource_lock = threading.Lock()

        # Monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._resource_history: List[Tuple[datetime, SystemResources]] = []

    def _detect_system_resources(self) -> SystemResources:
        """Detect available system resources"""
        try:
            # CPU
            total_cpu = psutil.cpu_count(logical=True)
            available_cpu = total_cpu if self.max_cpu_cores is None else min(total_cpu, self.max_cpu_cores)

            # Memory
            memory = psutil.virtual_memory()
            total_memory_mb = memory.total // (1024 * 1024)
            if self.max_memory_mb is None:
                # Use 80% of available memory
                available_memory_mb = int(memory.available * 0.8) // (1024 * 1024)
            else:
                available_memory_mb = min(total_memory_mb, self.max_memory_mb)

            # GPU
            gpu_available = False
            gpu_count = 0
            gpu_memory = []

            if self.enable_gpu:
                try:
                    if GPU_UTIL_AVAILABLE and GPUtil:
                        gpus = GPUtil.getGPUs()
                        gpu_count = len(gpus)
                        gpu_available = gpu_count > 0
                        gpu_memory = [gpu.memoryTotal for gpu in gpus]
                    else:
                        gpu_available = False
                        gpu_count = 0
                        gpu_memory = []
                except:
                    gpu_available = False
                    gpu_count = 0
                    gpu_memory = []

            # Disk
            disk = psutil.disk_usage('/')
            disk_space_mb = disk.free // (1024 * 1024)

            return SystemResources(
                total_cpu_cores=total_cpu,
                available_cpu_cores=available_cpu,
                total_memory_mb=total_memory_mb,
                available_memory_mb=available_memory_mb,
                gpu_available=gpu_available,
                gpu_count=gpu_count,
                gpu_memory_mb=gpu_memory,
                disk_space_mb=disk_space_mb
            )

        except Exception as e:
            self.logger.error(f"Failed to detect system resources: {e}")
            # Return minimal resources
            return SystemResources(
                total_cpu_cores=1,
                available_cpu_cores=1,
                total_memory_mb=1024,
                available_memory_mb=512,
                disk_space_mb=1024
            )

    def allocate_resources(self,
                          task_id: str,
                          requirements: ResourceRequirements) -> ResourceAllocation:
        """
        Allocate resources for a task

        Args:
            task_id: Task identifier
            requirements: Resource requirements

        Returns:
            Resource allocation

        Raises:
            ResourceError: If resources cannot be allocated
        """
        with self._resource_lock:
            if task_id in self._allocations:
                raise ResourceError(f"Resources already allocated for task {task_id}")

            try:
                allocation = self._current_resources.allocate(requirements)
                allocation.task_id = task_id
                self._allocations[task_id] = allocation

                self.logger.info(f"Allocated resources for task {task_id}: "
                               f"CPU={allocation.cpu_cores_allocated}, "
                               f"Memory={allocation.memory_mb_allocated}MB, "
                               f"GPU={allocation.gpu_allocated}")

                return allocation

            except Exception as e:
                raise ResourceError(f"Failed to allocate resources for task {task_id}: {e}")

    def release_resources(self, task_id: str) -> None:
        """
        Release resources allocated to a task

        Args:
            task_id: Task identifier
        """
        with self._resource_lock:
            if task_id not in self._allocations:
                self.logger.warning(f"No resources allocated for task {task_id}")
                return

            allocation = self._allocations[task_id]
            self._current_resources.release_allocation(allocation)
            del self._allocations[task_id]

            self.logger.info(f"Released resources for task {task_id}")

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        with self._resource_lock:
            return {
                'current_resources': {
                    'cpu_cores_available': self._current_resources.available_cpu_cores,
                    'memory_mb_available': self._current_resources.available_memory_mb,
                    'gpu_available': self._current_resources.gpu_available,
                    'gpu_count': self._current_resources.gpu_count,
                    'disk_space_mb': self._current_resources.disk_space_mb
                },
                'allocations': {
                    task_id: {
                        'cpu_cores': alloc.cpu_cores_allocated,
                        'memory_mb': alloc.memory_mb_allocated,
                        'gpu': alloc.gpu_allocated,
                        'allocated_at': alloc.allocated_at.isoformat()
                    }
                    for task_id, alloc in self._allocations.items()
                },
                'total_allocations': len(self._allocations)
            }

    def wait_for_resources(self,
                          requirements: ResourceRequirements,
                          timeout: int = 300) -> bool:
        """
        Wait for resources to become available

        Args:
            requirements: Required resources
            timeout: Maximum wait time in seconds

        Returns:
            True if resources became available, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._resource_lock:
                if self._current_resources.can_allocate(requirements):
                    return True

            time.sleep(1.0)  # Check every second

        return False

    def prioritize_tasks(self, task_requirements: Dict[str, ResourceRequirements]) -> List[str]:
        """
        Prioritize tasks based on resource requirements and availability

        Args:
            task_requirements: Dict mapping task IDs to requirements

        Returns:
            List of task IDs in priority order
        """
        # Sort by priority (higher first) and resource requirements (lower first)
        def sort_key(task_id: str) -> Tuple[int, int, int]:
            req = task_requirements[task_id]
            return (
                -req.priority,  # Higher priority first
                req.cpu_cores,  # Lower CPU requirement first
                req.memory_mb   # Lower memory requirement first
            )

        return sorted(task_requirements.keys(), key=sort_key)

    def get_resource_forecast(self, future_requirements: List[ResourceRequirements]) -> Dict[str, Any]:
        """
        Forecast resource usage for future tasks

        Args:
            future_requirements: List of future resource requirements

        Returns:
            Resource forecast
        """
        with self._resource_lock:
            forecast = []
            temp_resources = SystemResources(
                total_cpu_cores=self._current_resources.total_cpu_cores,
                available_cpu_cores=self._current_resources.available_cpu_cores,
                total_memory_mb=self._current_resources.total_memory_mb,
                available_memory_mb=self._current_resources.available_memory_mb,
                gpu_available=self._current_resources.gpu_available,
                gpu_count=self._current_resources.gpu_count,
                gpu_memory_mb=self._current_resources.gpu_memory_mb.copy(),
                disk_space_mb=self._current_resources.disk_space_mb
            )

            for i, req in enumerate(future_requirements):
                can_allocate = temp_resources.can_allocate(req)
                forecast.append({
                    'task_index': i,
                    'can_allocate': can_allocate,
                    'cpu_available_after': temp_resources.available_cpu_cores,
                    'memory_available_after': temp_resources.available_memory_mb
                })

                if can_allocate:
                    temp_resources.allocate(req)

            return {
                'forecast': forecast,
                'bottlenecks': self._identify_bottlenecks(future_requirements)
            }

    def _identify_bottlenecks(self, requirements: List[ResourceRequirements]) -> List[str]:
        """Identify potential resource bottlenecks"""
        bottlenecks = []

        total_cpu_needed = sum(req.cpu_cores for req in requirements)
        total_memory_needed = sum(req.memory_mb for req in requirements)
        gpu_needed = any(req.gpu_required for req in requirements)

        if total_cpu_needed > self._current_resources.total_cpu_cores:
            bottlenecks.append(f"CPU bottleneck: {total_cpu_needed} cores needed, "
                             f"{self._current_resources.total_cpu_cores} available")

        if total_memory_needed > self._current_resources.total_memory_mb:
            bottlenecks.append(f"Memory bottleneck: {total_memory_needed}MB needed, "
                             f"{self._current_resources.total_memory_mb}MB available")

        if gpu_needed and not self._current_resources.gpu_available:
            bottlenecks.append("GPU required but not available")

        return bottlenecks

    def start_monitoring(self):
        """Start resource monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")

    def _monitoring_loop(self):
        """Resource monitoring loop"""
        while self._monitoring_active:
            try:
                # Update current resources
                with self._resource_lock:
                    self._current_resources = self._detect_system_resources()
                    self._resource_history.append((datetime.now(), self._current_resources))

                    # Keep only last 100 entries
                    if len(self._resource_history) > 100:
                        self._resource_history = self._resource_history[-100:]

                # Log resource usage if monitoring enabled
                if feature_monitor.enable_monitoring:
                    with feature_monitor.monitor_operation(
                        "resource_monitoring",
                        cpu_available=self._current_resources.available_cpu_cores,
                        memory_available_mb=self._current_resources.available_memory_mb,
                        gpu_available=self._current_resources.gpu_available,
                        active_allocations=len(self._allocations)
                    ):
                        pass

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                break

    def get_resource_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get resource usage history"""
        with self._resource_lock:
            history = self._resource_history[-last_n:] if last_n > 0 else self._resource_history

            return [
                {
                    'timestamp': timestamp.isoformat(),
                    'cpu_available': resources.available_cpu_cores,
                    'memory_available_mb': resources.available_memory_mb,
                    'gpu_available': resources.gpu_available,
                    'allocations': len(self._allocations)
                }
                for timestamp, resources in history
            ]


# Global resource manager instance
resource_manager = ResourceManager()


__all__ = [
    "ResourceRequirements",
    "ResourceAllocation",
    "SystemResources",
    "ResourceError",
    "ResourceManager",
    "resource_manager"
]