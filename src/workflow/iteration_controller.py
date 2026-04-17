from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class WorkflowStep:
    """工作流步骤"""
    name: str
    description: str
    step_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """步骤执行结果"""
    step_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IterationController:
    """迭代控制器"""

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.95
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.history: List[Dict[str, Any]] = []
        self.current_iteration = 0

    def should_continue(self, quality_score: float = 0.0) -> bool:
        """判断是否应该继续迭代"""
        if self.current_iteration >= self.max_iterations:
            return False
        if quality_score >= self.convergence_threshold:
            return False
        return True

    def record_iteration(
        self,
        iteration: int,
        operations: List[Dict[str, Any]],
        quality_score: float
    ) -> None:
        """记录迭代信息"""
        self.history.append({
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "operations": operations,
            "quality_score": quality_score
        })
        self.current_iteration = iteration

    def record_step(self, step: WorkflowStep, result: StepResult) -> None:
        """记录步骤执行"""
        self.history.append({
            "step": step.name,
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "error": result.error,
            "duration": result.duration_seconds
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.history

    def get_current_iteration(self) -> int:
        """获取当前迭代次数"""
        return self.current_iteration

    def reset(self) -> None:
        """重置控制器"""
        self.history.clear()
        self.current_iteration = 0
