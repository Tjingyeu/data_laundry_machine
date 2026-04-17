from jinja2 import Template
from typing import Dict, Any, Optional
import os


class PromptManager:
    """Prompt 模板管理器"""

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            # 默认使用相对于 src 的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            templates_dir = os.path.join(current_dir, "templates")
        self.templates_dir = templates_dir
        self._cache: Dict[str, str] = {}

    def get_system_prompt(self) -> str:
        """获取系统级 Prompt"""
        return self._load_template("system_prompt.txt")

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """渲染 Prompt 模板"""
        template_content = self._load_template(f"{template_name}.txt")
        template = Template(template_content)
        return template.render(**context)

    def _load_template(self, filename: str) -> str:
        if filename in self._cache:
            return self._cache[filename]

        filepath = os.path.join(self.templates_dir, filename)
        if not os.path.exists(filepath):
            # 如果模板文件不存在，返回空字符串或默认模板
            return self._get_default_template(filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self._cache[filename] = content
        return content

    def _get_default_template(self, filename: str) -> str:
        """获取默认模板"""
        defaults = {
            "system_prompt.txt": DEFAULT_SYSTEM_PROMPT,
            "analysis_request.txt": DEFAULT_ANALYSIS_REQUEST,
            "operation_result.txt": DEFAULT_OPERATION_RESULT,
        }
        return defaults.get(filename, "")

    def clear_cache(self) -> None:
        """清空模板缓存"""
        self._cache.clear()


# 默认系统 Prompt
DEFAULT_SYSTEM_PROMPT = """You are a professional data cleaning expert Agent. Your task is to analyze data samples, identify data quality issues, and generate specific cleaning operation instructions to fix these issues.

## Your Capabilities
You can call the following tools to execute data cleaning operations:
- handle_missing_values: Handle missing values in columns
- deduplicate: Remove duplicate rows
- convert_type: Convert column data types
- normalize_format: Standardize data formats
- detect_anomalies: Detect anomalous values
- validate_data: Validate data quality
- finish_cleaning: Mark cleaning as complete

## Workflow
1. Analyze the provided data sample, identify all data quality issues
2. Create a cleaning plan, prioritize high-impact issues
3. Call tools one by one to execute cleaning operations
4. After completion, call finish_cleaning to mark the end

## Important Notes
- Always call one tool at a time, wait for execution results before deciding next steps
- Ensure all operations have explicit column names and parameters
- Before calling finish_cleaning, ensure major issues are resolved
- Provide clear reasoning for each operation you recommend
"""

# 默认分析请求 Prompt
DEFAULT_ANALYSIS_REQUEST = """Please analyze the following data sample, identify data quality issues and develop a cleaning strategy.

## Data Sample
```json
{{ data_sample }}
```

## Data Schema
```json
{{ data_schema }}
```

## Conversation History
{{ conversation_history }}

## Your Task
1. First analyze the data, identify all issues (missing values, duplicates, type errors, format inconsistencies, anomalies, etc.)
2. Create a cleaning plan, sorted by priority
3. Start calling tools to execute cleaning operations

Please start analyzing and executing cleaning.
"""

# 操作结果反馈 Prompt
DEFAULT_OPERATION_RESULT = """Operation "{{ operation_name }}" has been executed.

## Execution Result
- Success: {{ success }}
- Affected Rows: {{ affected_rows }}
- Details: {{ details }}

{% if warnings %}
## Warnings
{% for warning in warnings %}
- {{ warning }}
{% endfor %}
{% endif %}

Please analyze the result and decide next steps.
"""
