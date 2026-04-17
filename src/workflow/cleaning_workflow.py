from typing import Optional, Dict, Any, List
import pandas as pd
import logging

from .iteration_controller import IterationController, WorkflowStep, StepResult
from ..agent.data_cleaning_agent import DataCleaningAgent, DataCleaningAgentConfig

logger = logging.getLogger(__name__)


class CleaningWorkflow:
    """数据清洗工作流"""

    def __init__(
        self,
        agent: DataCleaningAgent,
        iteration_controller: Optional[IterationController] = None
    ):
        self.agent = agent
        self.controller = iteration_controller or IterationController()
        self.steps: List[WorkflowStep] = []
        self.pre_cleaning_steps: List[WorkflowStep] = []
        self.post_cleaning_steps: List[WorkflowStep] = []

    def add_pre_cleaning_step(self, step: WorkflowStep) -> None:
        """添加清洗前步骤"""
        self.pre_cleaning_steps.append(step)

    def add_post_cleaning_step(self, step: WorkflowStep) -> None:
        """添加清洗后步骤"""
        self.post_cleaning_steps.append(step)

    async def execute(
        self,
        initial_data: pd.DataFrame,
        pre_process: Optional[callable] = None,
        post_process: Optional[callable] = None
    ) -> Dict[str, Any]:
        """执行完整工作流"""
        current_data = initial_data.copy()
        results = {
            "pre_processing": [],
            "cleaning": None,
            "post_processing": [],
            "final_data": current_data,
            "success": True,
            "errors": []
        }

        # 前置处理步骤
        for step in self.pre_cleaning_steps:
            logger.info(f"Executing pre-processing step: {step.name}")
            try:
                step_result = await self._execute_step(step, current_data)
                results["pre_processing"].append(step_result)
                if step_result.data is not None:
                    current_data = step_result.data
            except Exception as e:
                logger.error(f"Pre-processing step failed: {step.name}, error: {e}")
                results["errors"].append(f"Pre-processing {step.name}: {str(e)}")

        # 自定义预清洗处理
        if pre_process:
            try:
                current_data = pre_process(current_data)
            except Exception as e:
                logger.error(f"Pre-process callback failed: {e}")
                results["errors"].append(f"Pre-process callback: {str(e)}")

        # 核心清洗
        try:
            logger.info("Starting core cleaning process")
            cleaning_result = await self.agent.clean(current_data)
            results["cleaning"] = {
                "total_rows": cleaning_result.total_rows,
                "cleaned_rows": cleaning_result.cleaned_rows,
                "operations": cleaning_result.operations_applied,
                "quality_report": cleaning_result.data_quality_report
            }
            current_data = self.agent.data_sample

            # 记录迭代
            self.controller.record_iteration(
                iteration=self.controller.current_iteration + 1,
                operations=cleaning_result.operations_applied,
                quality_score=cleaning_result.data_quality_report.get("score", 0.5)
            )

        except Exception as e:
            logger.error(f"Cleaning failed: {e}")
            results["errors"].append(f"Cleaning: {str(e)}")
            results["success"] = False

        # 自定义后清洗处理
        if post_process:
            try:
                current_data = post_process(current_data)
            except Exception as e:
                logger.error(f"Post-process callback failed: {e}")
                results["errors"].append(f"Post-process callback: {str(e)}")

        # 后置处理步骤
        for step in self.post_cleaning_steps:
            logger.info(f"Executing post-processing step: {step.name}")
            try:
                step_result = await self._execute_step(step, current_data)
                results["post_processing"].append(step_result)
                if step_result.data is not None:
                    current_data = step_result.data
            except Exception as e:
                logger.error(f"Post-processing step failed: {step.name}, error: {e}")
                results["errors"].append(f"Post-processing {step.name}: {str(e)}")

        results["final_data"] = current_data
        return results

    async def _execute_step(
        self,
        step: WorkflowStep,
        data: pd.DataFrame
    ) -> StepResult:
        """执行单个步骤"""
        from datetime import datetime
        start_time = datetime.now()

        # 根据步骤类型执行
        if step.step_type == "validate":
            # 验证步骤
            is_valid = self._validate_data(data, step)
            duration = (datetime.now() - start_time).total_seconds()

            return StepResult(
                step_name=step.name,
                success=is_valid,
                duration_seconds=duration,
                metadata=step.parameters
            )

        elif step.step_type == "transform":
            # 转换步骤
            transformed = self._transform_data(data, step)
            duration = (datetime.now() - start_time).total_seconds()

            return StepResult(
                step_name=step.name,
                success=True,
                data=transformed,
                duration_seconds=duration,
                metadata=step.parameters
            )

        else:
            duration = (datetime.now() - start_time).total_seconds()
            return StepResult(
                step_name=step.name,
                success=False,
                error=f"Unknown step type: {step.step_type}",
                duration_seconds=duration
            )

    def _validate_data(self, data: pd.DataFrame, step: WorkflowStep) -> bool:
        """验证数据"""
        rules = step.parameters.get("rules", {})
        errors = []

        for rule, value in rules.items():
            if rule == "min_rows" and len(data) < value:
                errors.append(f"Row count {len(data)} < {value}")
            elif rule == "max_nulls":
                null_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
                if null_pct > value:
                    errors.append(f"Null percentage {null_pct} > {value}")

        return len(errors) == 0

    def _transform_data(self, data: pd.DataFrame, step: WorkflowStep) -> pd.DataFrame:
        """转换数据"""
        transform_type = step.parameters.get("type")

        if transform_type == "rename_columns":
            mapping = step.parameters.get("mapping", {})
            return data.rename(columns=mapping)

        elif transform_type == "drop_columns":
            columns = step.parameters.get("columns", [])
            return data.drop(columns=columns)

        elif transform_type == "select_columns":
            columns = step.parameters.get("columns", [])
            return data[columns]

        return data

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "total_iterations": self.controller.get_current_iteration(),
            "max_iterations": self.controller.max_iterations,
            "convergence_threshold": self.controller.convergence_threshold,
            "history": self.controller.get_history()
        }
