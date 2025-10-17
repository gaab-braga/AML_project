# src/orchestration/dag_engine.py
"""
DAG Engine for parallel task execution
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import threading

from .task_definition import DAG, Task, TaskResult
from ..utils import logger
from ..features.metrics import feature_monitor


@dataclass
class ExecutionContext:
    """Context for task execution"""
    task_id: str
    start_time: float
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DAGEngine:
    """
    Directed Acyclic Graph execution engine

    Handles parallel execution of tasks with dependency management
    """

    def __init__(self,
                 max_workers: int = 4,
                 enable_monitoring: bool = True,
                 timeout: int = 3600):
        """
        Initialize DAG Engine

        Args:
            max_workers: Maximum number of parallel workers
            enable_monitoring: Whether to enable performance monitoring
            timeout: Default timeout for task execution (seconds)
        """
        self.max_workers = max_workers
        self.enable_monitoring = enable_monitoring
        self.default_timeout = timeout
        self.logger = logger.getChild("dag_engine")

        # Execution state
        self._execution_lock = threading.Lock()
        self._running_tasks: Set[str] = set()
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._failed_tasks: Dict[str, Exception] = {}

    def execute_dag(self,
                   dag: DAG,
                   target_tasks: Optional[List[str]] = None,
                   execution_mode: str = "parallel") -> Dict[str, TaskResult]:
        """
        Execute a DAG

        Args:
            dag: DAG to execute
            target_tasks: Specific tasks to execute (None = all tasks)
            execution_mode: "parallel" or "sequential"

        Returns:
            Dictionary mapping task IDs to execution results
        """
        start_time = time.time()

        # Validate DAG
        validation = dag.validate()
        if not validation['valid']:
            raise ValueError(f"Invalid DAG: {validation['issues']}")

        self.logger.info(f"Starting DAG execution: {dag.name} "
                        f"(mode: {execution_mode}, workers: {self.max_workers})")

        # Determine execution order
        execution_order = dag.get_execution_order()

        # Filter to target tasks if specified
        if target_tasks:
            execution_order = self._filter_execution_order(execution_order, target_tasks)

        # Execute based on mode
        if execution_mode == "parallel":
            results = self._execute_parallel(dag, execution_order)
        elif execution_mode == "sequential":
            results = self._execute_sequential(dag, execution_order)
        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

        execution_time = time.time() - start_time
        self.logger.info(f"DAG execution completed in {execution_time:.2f}s")

        # Log execution summary
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful

        self.logger.info(f"Execution summary: {successful} successful, {failed} failed")

        if self.enable_monitoring:
            with feature_monitor.monitor_operation(
                "dag_execution",
                dag_name=dag.name,
                execution_mode=execution_mode,
                total_tasks=len(results),
                successful_tasks=successful,
                failed_tasks=failed,
                execution_time=execution_time
            ):
                pass

        return results

    async def execute_dag_async(self,
                               dag: DAG,
                               target_tasks: Optional[List[str]] = None) -> Dict[str, TaskResult]:
        """
        Execute DAG asynchronously

        Args:
            dag: DAG to execute
            target_tasks: Specific tasks to execute

        Returns:
            Dictionary mapping task IDs to execution results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.execute_dag,
            dag,
            target_tasks,
            "parallel"
        )

    def _execute_parallel(self, dag: DAG, execution_order: List[List[str]]) -> Dict[str, TaskResult]:
        """Execute tasks in parallel"""
        results = {}

        for level in execution_order:
            level_results = self._execute_level_parallel(dag, level)
            results.update(level_results)

            # Check for failures that should stop execution
            failed_tasks = [task_id for task_id, result in level_results.items() if not result.success]
            if failed_tasks:
                self.logger.warning(f"Level execution had failures: {failed_tasks}")
                # Continue with other levels but log warnings

        return results

    def _execute_level_parallel(self, dag: DAG, level_tasks: List[str]) -> Dict[str, TaskResult]:
        """Execute a single level of tasks in parallel"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks in this level
            future_to_task = {}
            for task_id in level_tasks:
                task = dag.get_task(task_id)
                future = executor.submit(self._execute_task_with_retry, task, dag)
                future_to_task[future] = task_id

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    results[task_id] = result
                except Exception as e:
                    self.logger.error(f"Task {task_id} execution failed: {e}")
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        success=False,
                        error=str(e),
                        execution_time=0.0
                    )

        return results

    def _execute_sequential(self, dag: DAG, execution_order: List[List[str]]) -> Dict[str, TaskResult]:
        """Execute tasks sequentially"""
        results = {}

        for level in execution_order:
            for task_id in level:
                task = dag.get_task(task_id)
                result = self._execute_task_with_retry(task, dag)
                results[task_id] = result

        return results

    def _execute_task_with_retry(self, task: Task, dag: DAG) -> TaskResult:
        """Execute a task with retry logic"""
        context = ExecutionContext(
            task_id=task.id,
            start_time=time.time(),
            timeout=task.timeout or self.default_timeout,
            max_retries=task.retry_count
        )

        last_error = None

        for attempt in range(context.max_retries + 1):
            try:
                context.retry_count = attempt
                result = self._execute_task_single(task, context, dag)

                if result.success:
                    return result
                else:
                    last_error = Exception(result.error or "Task failed")

            except Exception as e:
                last_error = e
                self.logger.warning(f"Task {task.id} attempt {attempt + 1} failed: {e}")

                if attempt < context.max_retries:
                    # Exponential backoff
                    backoff_time = 2 ** attempt
                    self.logger.info(f"Retrying task {task.id} in {backoff_time}s")
                    time.sleep(backoff_time)

        # All retries exhausted
        execution_time = time.time() - context.start_time
        return TaskResult(
            task_id=task.id,
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            execution_time=execution_time,
            retry_count=context.retry_count
        )

    def _execute_task_single(self, task: Task, context: ExecutionContext, dag: DAG) -> TaskResult:
        """Execute a single task"""
        start_time = time.time()

        with self._execution_lock:
            self._running_tasks.add(task.id)

        try:
            # Prepare inputs from dependencies
            inputs = self._prepare_task_inputs(task, dag, context)

            # Execute task
            self.logger.info(f"Executing task: {task.id} ({task.name})")

            if self.enable_monitoring:
                with feature_monitor.monitor_operation(
                    f"task_execution_{task.id}",
                    task_name=task.name,
                    task_priority=task.priority,
                    input_count=len(inputs)
                ) as metrics:
                    # Execute with timeout
                    if task.timeout:
                        result = self._execute_with_timeout(task.function, inputs, task.timeout)
                    else:
                        result = task.function(**inputs)

                    metrics.output_shape = getattr(result, 'shape', None) if hasattr(result, '__len__') else None
                    metrics.features_created = getattr(result, 'shape', [0, 0])[1] if hasattr(result, 'shape') else 0

            else:
                if task.timeout:
                    result = self._execute_with_timeout(task.function, inputs, task.timeout)
                else:
                    result = task.function(**inputs)

            execution_time = time.time() - start_time

            # Create success result
            task_result = TaskResult(
                task_id=task.id,
                success=True,
                outputs={'result': result} if not isinstance(result, dict) else result,
                execution_time=execution_time,
                retry_count=context.retry_count,
                metadata={
                    'attempt': context.retry_count + 1,
                    'input_count': len(inputs)
                }
            )

            # Store completed task result
            with self._execution_lock:
                self._completed_tasks[task.id] = task_result

            self.logger.info(f"Task {task.id} completed successfully in {execution_time:.2f}s")
            return task_result

            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                retry_count=context.retry_count
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task {task.id} failed: {str(e)}"

            # Store failed task
            with self._execution_lock:
                self._failed_tasks[task.id] = e

            self.logger.error(error_msg, extra={"extra_fields": {
                "task_id": task.id,
                "task_name": task.name,
                "execution_time": execution_time,
                "error": str(e)
            }})

            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                retry_count=context.retry_count
            )

        finally:
            with self._execution_lock:
                self._running_tasks.discard(task.id)

    def _prepare_task_inputs(self, task: Task, dag: DAG, context: ExecutionContext) -> Dict[str, Any]:
        """Prepare inputs for task execution"""
        inputs = task.inputs.copy()

        # Add outputs from dependencies
        for dep_id in task.dependencies:
            if dep_id in self._completed_tasks:
                dep_result = self._completed_tasks[dep_id]
                if dep_result.success and dep_result.outputs:
                    # Merge dependency outputs into inputs
                    inputs.update(dep_result.outputs)
            else:
                raise ValueError(f"Dependency {dep_id} not completed for task {task.id}")

        return inputs

    def _execute_with_timeout(self, func: callable, args: Dict[str, Any], timeout: int):
        """Execute function with timeout"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task execution timed out after {timeout}s")

        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            return func(**args)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _filter_execution_order(self,
                               execution_order: List[List[str]],
                               target_tasks: List[str]) -> List[List[str]]:
        """Filter execution order to only include necessary tasks"""
        target_set = set(target_tasks)
        required_tasks = set()

        # Find all tasks required for targets
        def add_dependencies(task_id: str):
            if task_id in required_tasks:
                return
            required_tasks.add(task_id)
            # Add all dependencies recursively
            for dep_id in execution_order:  # This is wrong, need to fix
                pass  # TODO: Implement proper dependency traversal

        for task_id in target_tasks:
            add_dependencies(task_id)

        # Filter execution order
        filtered_order = []
        for level in execution_order:
            filtered_level = [task_id for task_id in level if task_id in required_tasks]
            if filtered_level:
                filtered_order.append(filtered_level)

        return filtered_order

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        with self._execution_lock:
            return {
                'running_tasks': list(self._running_tasks),
                'completed_tasks': len(self._completed_tasks),
                'failed_tasks': len(self._failed_tasks),
                'total_active': len(self._running_tasks)
            }

    def cancel_execution(self) -> None:
        """Cancel current execution"""
        self.logger.info("Cancelling DAG execution")
        # TODO: Implement proper cancellation logic
        with self._execution_lock:
            self._running_tasks.clear()


__all__ = [
    "ExecutionContext",
    "DAGEngine"
]