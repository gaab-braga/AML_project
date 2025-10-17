# src/orchestration/task_definition.py
"""
Task Definition System for AML Pipeline Orchestration
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional, Set
from datetime import datetime
import uuid

from ..utils import logger


@dataclass
class Task:
    """
    Represents a single task in the pipeline

    Attributes:
        id: Unique identifier for the task
        name: Human-readable name
        function: Callable to execute
        dependencies: List of task IDs this task depends on
        inputs: Dictionary of input parameters
        outputs: Dictionary of expected outputs
        timeout: Maximum execution time in seconds
        retry_count: Number of retry attempts on failure
        priority: Execution priority (higher = more important)
        resource_requirements: CPU/memory requirements
        metadata: Additional task metadata
    """
    id: str
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    priority: int = 1
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Validate task configuration after initialization"""
        if not self.id:
            raise ValueError("Task id cannot be empty")
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if not callable(self.function):
            raise ValueError("Task function must be callable")

        # Set default resource requirements
        if not self.resource_requirements:
            self.resource_requirements = {
                'cpu_cores': 1,
                'memory_mb': 512,
                'gpu_required': False
            }

    def get_execution_key(self) -> str:
        """Get unique execution key for this task"""
        return f"{self.id}_{self.task_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'dependencies': self.dependencies,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'priority': self.priority,
            'resource_requirements': self.resource_requirements,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'task_id': self.task_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], function: Callable) -> 'Task':
        """Create task from dictionary"""
        # Remove function from data since it's provided separately
        task_data = data.copy()
        task_data['function'] = function
        if 'created_at' in task_data:
            task_data['created_at'] = datetime.fromisoformat(task_data['created_at'])
        return cls(**task_data)


@dataclass
class TaskResult:
    """
    Result of task execution

    Attributes:
        task_id: ID of the executed task
        success: Whether execution was successful
        outputs: Task outputs (if successful)
        error: Error message (if failed)
        execution_time: Time taken to execute
        retry_count: Number of retries performed
        metadata: Additional execution metadata
    """
    task_id: str
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'outputs': self.outputs,
            'error': self.error,
            'execution_time': self.execution_time,
            'retry_count': self.retry_count,
            'metadata': self.metadata,
            'executed_at': self.executed_at.isoformat()
        }


class DAG:
    """
    Directed Acyclic Graph for task orchestration

    Manages task dependencies and execution order
    """

    def __init__(self, name: str = "aml_pipeline"):
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.reverse_dependencies: Dict[str, List[str]] = {}
        self.logger = logger.getChild(f"dag.{name}")

    def add_task(self, task: Task) -> None:
        """
        Add a task to the DAG

        Args:
            task: Task to add

        Raises:
            ValueError: If task ID already exists or creates a cycle
        """
        if task.id in self.tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")

        # Validate dependencies exist
        for dep_id in task.dependencies:
            if dep_id not in self.tasks and dep_id != task.id:
                raise ValueError(f"Dependency '{dep_id}' not found for task '{task.id}'")

        # Check for cycles
        if self._would_create_cycle(task.id, task.dependencies):
            raise ValueError(f"Adding task '{task.id}' would create a cycle")

        # Add task
        self.tasks[task.id] = task
        self.dependencies[task.id] = task.dependencies.copy()

        # Update reverse dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.reverse_dependencies:
                self.reverse_dependencies[dep_id] = []
            self.reverse_dependencies[dep_id].append(task.id)

        self.logger.info(f"Added task: {task.id} ({task.name})")

    def remove_task(self, task_id: str) -> None:
        """
        Remove a task from the DAG

        Args:
            task_id: ID of task to remove
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")

        # Remove from dependencies
        del self.dependencies[task_id]

        # Remove from reverse dependencies
        for dep_list in self.reverse_dependencies.values():
            if task_id in dep_list:
                dep_list.remove(task_id)

        # Remove task
        del self.tasks[task_id]

        self.logger.info(f"Removed task: {task_id}")

    def get_task(self, task_id: str) -> Task:
        """Get task by ID"""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        return self.tasks[task_id]

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in the DAG"""
        return list(self.tasks.values())

    def get_roots(self) -> List[Task]:
        """Get tasks with no dependencies (roots)"""
        return [task for task in self.tasks.values() if not task.dependencies]

    def get_leaves(self) -> List[Task]:
        """Get tasks with no dependents (leaves)"""
        return [task for task in self.tasks.values()
                if task.id not in self.reverse_dependencies]

    def get_execution_order(self) -> List[List[str]]:
        """
        Get execution order using topological sort

        Returns:
            List of lists, where each inner list contains tasks that can be
            executed in parallel
        """
        # Kahn's algorithm for topological sorting
        in_degree = {task_id: len(self.dependencies.get(task_id, [])) for task_id in self.tasks.keys()}
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            level = []
            next_queue = []

            for task_id in queue:
                level.append(task_id)

                # Decrease in-degree of dependents
                for dependent in self.reverse_dependencies.get(task_id, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)

            if level:
                result.append(level)
            queue = next_queue

        # Check for cycles
        processed_tasks = set()
        for level in result:
            for task_id in level:
                if task_id in processed_tasks:
                    raise ValueError(f"Task {task_id} appears multiple times in execution order")
                processed_tasks.add(task_id)

        if len(processed_tasks) != len(self.tasks):
            remaining = set(self.tasks.keys()) - processed_tasks
            raise ValueError(f"Cycle detected in DAG. Remaining tasks: {remaining}")

        return result

    def validate(self) -> Dict[str, Any]:
        """
        Validate DAG structure

        Returns:
            Validation results
        """
        issues = []
        warnings = []

        # Check for missing dependencies
        for task_id, deps in self.dependencies.items():
            for dep_id in deps:
                if dep_id not in self.tasks:
                    issues.append(f"Task '{task_id}' depends on missing task '{dep_id}'")

        # Check for cycles
        try:
            self.get_execution_order()
        except ValueError as e:
            issues.append(str(e))

        # Check for isolated tasks
        connected_tasks = set()
        for task_id in self.tasks:
            connected_tasks.add(task_id)
            connected_tasks.update(self.dependencies.get(task_id, []))
            connected_tasks.update(self.reverse_dependencies.get(task_id, []))

        isolated = set(self.tasks.keys()) - connected_tasks
        if isolated:
            warnings.append(f"Isolated tasks found: {isolated}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'task_count': len(self.tasks),
            'root_count': len(self.get_roots()),
            'leaf_count': len(self.get_leaves())
        }

    def _would_create_cycle(self, task_id: str, dependencies: List[str]) -> bool:
        """Check if adding a task would create a cycle"""
        # Simple cycle detection: check if any dependency depends on this task
        visited = set()

        def has_path_to_task(current_id: str) -> bool:
            if current_id in visited:
                return False
            visited.add(current_id)

            for dep_id in self.dependencies.get(current_id, []):
                if dep_id == task_id:
                    return True
                if has_path_to_task(dep_id):
                    return True
            return False

        for dep_id in dependencies:
            if has_path_to_task(dep_id):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert DAG to dictionary for serialization"""
        return {
            'name': self.name,
            'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            'dependencies': self.dependencies.copy()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get DAG statistics"""
        execution_order = self.get_execution_order()
        max_parallelism = max(len(level) for level in execution_order) if execution_order else 0

        return {
            'name': self.name,
            'total_tasks': len(self.tasks),
            'total_dependencies': sum(len(deps) for deps in self.dependencies.values()),
            'execution_levels': len(execution_order),
            'max_parallelism': max_parallelism,
            'roots': len(self.get_roots()),
            'leaves': len(self.get_leaves()),
            'average_dependencies': sum(len(deps) for deps in self.dependencies.values()) / len(self.tasks) if self.tasks else 0
        }


# Convenience functions
def create_task(id: str,
                name: str,
                function: Callable,
                dependencies: List[str] = None,
                **kwargs) -> Task:
    """Convenience function to create a task"""
    return Task(
        id=id,
        name=name,
        function=function,
        dependencies=dependencies or [],
        **kwargs
    )


__all__ = [
    "Task",
    "TaskResult",
    "DAG",
    "create_task"
]