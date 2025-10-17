# src/orchestration/parallel_executor.py
"""
Parallel Execution Engine for AML Pipeline
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import threading
import psutil
import os

from .task_definition import Task, TaskResult
from ..utils import logger
from ..features.metrics import feature_monitor


@dataclass
class WorkerStats:
    """Statistics for a worker"""
    worker_id: str
    tasks_completed: int = 0
    total_execution_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_since: datetime = field(default_factory=datetime.now)


class ParallelExecutor:
    """
    Parallel executor for running tasks concurrently

    Supports both thread-based and process-based parallelism
    """

    def __init__(self,
                 max_workers: int = None,
                 executor_type: str = "thread",  # "thread" or "process"
                 enable_monitoring: bool = True,
                 worker_timeout: int = 300):
        """
        Initialize parallel executor

        Args:
            max_workers: Maximum number of workers (default: CPU count)
            executor_type: "thread" or "process"
            enable_monitoring: Enable performance monitoring
            worker_timeout: Timeout for individual task execution
        """
        self.max_workers = max_workers or os.cpu_count()
        self.executor_type = executor_type
        self.enable_monitoring = enable_monitoring
        self.worker_timeout = worker_timeout

        self.logger = logger.getChild("parallel_executor")

        # Worker management
        self._workers: Dict[str, WorkerStats] = {}
        self._active_workers: Set[str] = set()
        self._worker_lock = threading.Lock()

        # Resource monitoring
        self._resource_monitor_thread: Optional[threading.Thread] = None
        self._monitoring_active = False

    def execute_tasks(self,
                     tasks: List[Task],
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, TaskResult]:
        """
        Execute multiple tasks in parallel

        Args:
            tasks: List of tasks to execute
            context: Shared execution context

        Returns:
            Dictionary mapping task IDs to results
        """
        if not tasks:
            return {}

        start_time = time.time()
        context = context or {}

        self.logger.info(f"Starting parallel execution of {len(tasks)} tasks "
                        f"(workers: {self.max_workers}, type: {self.executor_type})")

        # Start resource monitoring
        if self.enable_monitoring:
            self._start_resource_monitoring()

        try:
            # Choose executor type
            if self.executor_type == "process":
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor

            results = {}

            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {}
                for task in tasks:
                    future = executor.submit(self._execute_task_wrapper, task, context)
                    future_to_task[future] = task.id

                # Collect results as they complete
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    try:
                        result = future.result(timeout=self.worker_timeout)
                        results[task_id] = result

                        # Update worker stats
                        self._update_worker_stats(result)

                    except Exception as e:
                        self.logger.error(f"Task {task_id} execution failed: {e}")
                        results[task_id] = TaskResult(
                            task_id=task_id,
                            success=False,
                            error=str(e),
                            execution_time=0.0
                        )

            execution_time = time.time() - start_time
            self.logger.info(f"Parallel execution completed in {execution_time:.2f}s")

            # Log performance metrics
            if self.enable_monitoring:
                successful = sum(1 for r in results.values() if r.success)
                failed = len(results) - successful

                with feature_monitor.monitor_operation(
                    "parallel_execution",
                    executor_type=self.executor_type,
                    max_workers=self.max_workers,
                    total_tasks=len(tasks),
                    successful_tasks=successful,
                    failed_tasks=failed,
                    execution_time=execution_time
                ):
                    pass

            return results

        finally:
            # Stop resource monitoring
            if self.enable_monitoring:
                self._stop_resource_monitoring()

    async def execute_tasks_async(self,
                                 tasks: List[Task],
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, TaskResult]:
        """
        Execute tasks asynchronously

        Args:
            tasks: List of tasks to execute
            context: Shared execution context

        Returns:
            Dictionary mapping task IDs to results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.execute_tasks,
            tasks,
            context
        )

    def _execute_task_wrapper(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        """Wrapper for task execution with monitoring"""
        worker_id = f"worker_{threading.current_thread().ident}"

        # Register worker
        with self._worker_lock:
            if worker_id not in self._workers:
                self._workers[worker_id] = WorkerStats(worker_id=worker_id)
            self._active_workers.add(worker_id)

        try:
            start_time = time.time()

            # Execute task
            result = self._execute_task(task, context)

            execution_time = time.time() - start_time

            # Update result with execution time
            result.execution_time = execution_time

            return result

        finally:
            # Unregister worker
            with self._worker_lock:
                self._active_workers.discard(worker_id)

    def _execute_task(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        """Execute a single task"""
        try:
            self.logger.debug(f"Executing task: {task.id} ({task.name})")

            # Merge context with task inputs
            inputs = {**context, **task.inputs}

            # Execute task function
            if task.timeout:
                result = self._execute_with_timeout(task.function, inputs, task.timeout)
            else:
                result = task.function(**inputs)

            # Create success result
            return TaskResult(
                task_id=task.id,
                success=True,
                outputs={'result': result} if not isinstance(result, dict) else result,
                metadata={
                    'worker_type': self.executor_type,
                    'input_count': len(inputs)
                }
            )

        except Exception as e:
            error_msg = f"Task {task.id} failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                metadata={'worker_type': self.executor_type}
            )

    def _execute_with_timeout(self, func: Callable, kwargs: Dict[str, Any], timeout: int):
        """Execute function with timeout using asyncio"""
        async def execute_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run in thread pool to avoid blocking
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: func(**kwargs)
                )
            finally:
                loop.close()

        try:
            return asyncio.run(asyncio.wait_for(execute_async(), timeout=timeout))
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task execution timed out after {timeout}s")

    def _update_worker_stats(self, result: TaskResult):
        """Update worker statistics"""
        worker_id = f"worker_{threading.current_thread().ident}"

        with self._worker_lock:
            if worker_id in self._workers:
                stats = self._workers[worker_id]
                stats.tasks_completed += 1
                stats.total_execution_time += result.execution_time

    def _start_resource_monitoring(self):
        """Start resource monitoring thread"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_loop,
            daemon=True
        )
        self._resource_monitor_thread.start()

    def _stop_resource_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring_active = False
        if self._resource_monitor_thread:
            self._resource_monitor_thread.join(timeout=1.0)

    def _resource_monitor_loop(self):
        """Resource monitoring loop"""
        while self._monitoring_active:
            try:
                # Update worker stats
                with self._worker_lock:
                    for worker_id in self._active_workers:
                        if worker_id in self._workers:
                            stats = self._workers[worker_id]
                            try:
                                stats.cpu_usage = psutil.cpu_percent(interval=None)
                                stats.memory_usage = psutil.virtual_memory().percent
                            except:
                                pass  # Ignore monitoring errors

                time.sleep(1.0)  # Update every second

            except Exception as e:
                self.logger.debug(f"Resource monitoring error: {e}")
                break

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        with self._worker_lock:
            return {
                'total_workers': len(self._workers),
                'active_workers': len(self._active_workers),
                'worker_details': {
                    worker_id: {
                        'tasks_completed': stats.tasks_completed,
                        'total_execution_time': stats.total_execution_time,
                        'avg_execution_time': (
                            stats.total_execution_time / stats.tasks_completed
                            if stats.tasks_completed > 0 else 0
                        ),
                        'cpu_usage': stats.cpu_usage,
                        'memory_usage': stats.memory_usage,
                        'active_since': stats.active_since.isoformat()
                    }
                    for worker_id, stats in self._workers.items()
                }
            }

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk.percent,
                'disk_free_mb': disk.free / (1024 * 1024)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get resource usage: {e}")
            return {}

    def scale_workers(self, target_utilization: float = 0.8) -> int:
        """
        Dynamically scale number of workers based on utilization

        Args:
            target_utilization: Target CPU utilization (0.0 to 1.0)

        Returns:
            New number of workers
        """
        current_usage = self.get_resource_usage()
        cpu_percent = current_usage.get('cpu_percent', 0) / 100.0

        if cpu_percent > target_utilization * 1.2:  # Overutilized
            new_workers = min(self.max_workers, self.max_workers + 1)
        elif cpu_percent < target_utilization * 0.8:  # Underutilized
            new_workers = max(1, self.max_workers - 1)
        else:
            new_workers = self.max_workers

        if new_workers != self.max_workers:
            self.logger.info(f"Scaling workers from {self.max_workers} to {new_workers} "
                           f"(CPU: {cpu_percent:.1%})")
            self.max_workers = new_workers

        return new_workers


__all__ = [
    "WorkerStats",
    "ParallelExecutor"
]