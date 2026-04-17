import click
import yaml
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional

from ..ai_providers import ProviderFactory
from ..agent import DataCleaningAgent, DataCleaningAgentConfig
from ..prompts import PromptManager
from ..workflow import CleaningWorkflow, IterationController
from ..data_source import CSVAdapter, JSONAdapter


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Data Laundry Machine - AI-powered data cleaning agent"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--provider", "-p", default="openai", type=click.Choice(["openai", "anthropic"]))
@click.option("--api-key", "-k", envvar="OPENAI_API_KEY", help="API key for AI provider")
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--max-iterations", "-i", default=5, type=int)
@click.option("--sample-size", "-s", default=100, type=int)
@click.option("--config", "-c", type=click.Path(), help="Configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def clean(
    input_file: str,
    output_file: str,
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
    max_iterations: int,
    sample_size: int,
    config: Optional[str],
    verbose: bool
):
    """Clean data using AI-powered agent"""
    import logging

    # 配置日志
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    # 加载配置
    config_data = {}
    if config and os.path.exists(config):
        with open(config) as f:
            config_data = yaml.safe_load(f)

    # 获取 API key
    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        click.echo("Error: API key is required. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable, or use --api-key option", err=True)
        sys.exit(1)

    # 确定模型
    if not model:
        if provider == "openai":
            model = config_data.get("ai", {}).get("model", "gpt-4o")
        else:
            model = config_data.get("ai", {}).get("model", "claude-3-5-sonnet-20241022")

    # 读取输入数据
    click.echo(f"Reading input file: {input_file}")
    input_path = Path(input_file)

    try:
        if input_path.suffix.lower() == ".csv":
            adapter = CSVAdapter(input_file)
        elif input_path.suffix.lower() == ".json":
            adapter = JSONAdapter(input_file)
        else:
            click.echo(f"Error: Unsupported file format: {input_path.suffix}", err=True)
            sys.exit(1)

        raw_data = adapter.read()
        click.echo(f"Loaded {len(raw_data)} rows, {len(raw_data.columns)} columns")

    except Exception as e:
        click.echo(f"Error reading input file: {e}", err=True)
        sys.exit(1)

    # 创建 AI 供应商
    click.echo(f"Creating {provider} provider with model {model}")
    ai_provider = ProviderFactory.create(
        provider_name=provider,
        api_key=api_key,
        model=model
    )

    # 创建 Agent
    prompt_manager = PromptManager()
    agent_config = DataCleaningAgentConfig(
        max_iterations=max_iterations,
        sample_size=sample_size
    )

    agent = DataCleaningAgent(
        ai_provider=ai_provider,
        prompt_manager=prompt_manager,
        config=agent_config
    )

    # 执行清洗
    click.echo("Starting AI-powered data cleaning...")
    import asyncio

    async def run_cleaning():
        return await agent.clean(raw_data)

    result = asyncio.run(run_cleaning())

    # 保存结果
    click.echo(f"Cleaning complete! Applied {len(result.operations_applied)} operations")

    # 导出数据
    output_path = Path(output_file)
    try:
        if output_path.suffix.lower() == ".csv":
            agent.data_sample.to_csv(output_file, index=False)
        elif output_path.suffix.lower() == ".json":
            agent.data_sample.to_json(output_file, orient="records", indent=2)
        click.echo(f"Saved cleaned data to: {output_file}")
    except Exception as e:
        click.echo(f"Error saving output file: {e}", err=True)
        sys.exit(1)

    # 显示质量报告
    click.echo("\n=== Data Quality Report ===")
    report = result.data_quality_report
    click.echo(f"Total rows: {report.get('total_rows', 'N/A')}")
    click.echo(f"Total columns: {report.get('total_columns', 'N/A')}")
    click.echo(f"Operations applied: {len(result.operations_applied)}")

    if verbose:
        click.echo("\n=== Operations Log ===")
        for op in result.operations_applied:
            click.echo(f"  - {op.get('operation_type')}: {op.get('affected_rows')} rows affected")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def analyze(input_file: str, format: str):
    """Analyze data quality"""
    import asyncio

    input_path = Path(input_file)

    try:
        if input_path.suffix.lower() == ".csv":
            adapter = CSVAdapter(input_file)
        elif input_path.suffix.lower() == ".json":
            adapter = JSONAdapter(input_file)
        else:
            click.echo(f"Error: Unsupported file format: {input_path.suffix}", err=True)
            sys.exit(1)

        raw_data = adapter.read()
        report = adapter.get_quality_report()

        if format == "json":
            import json
            click.echo(json.dumps(report, indent=2, default=str))
        else:
            click.echo("=== Data Quality Report ===")
            click.echo(f"Total rows: {report.get('total_rows')}")
            click.echo(f"Total columns: {report.get('total_columns')}")
            click.echo(f"Duplicate rows: {report.get('duplicate_rows')}")

            click.echo("\n=== Missing Values ===")
            for col, stats in report.get("missing_values", {}).items():
                if stats.get("count", 0) > 0:
                    click.echo(f"  {col}: {stats.get('count')} ({stats.get('percentage')}%)")

            click.echo("\n=== Column Types ===")
            for col, stats in report.get("column_stats", {}).items():
                click.echo(f"  {col}: {stats.get('dtype')} (unique: {stats.get('unique_count')})")

    except Exception as e:
        click.echo(f"Error analyzing file: {e}", err=True)
        sys.exit(1)


@cli.command()
def providers():
    """List available AI providers"""
    available = ProviderFactory.available_providers()
    click.echo("Available AI providers:")
    for p in available:
        click.echo(f"  - {p}")


@cli.command()
def init():
    """Initialize a new data laundry project"""
    config_content = """ai:
  provider: "openai"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o"
  temperature: 0.1
  max_tokens: 4096

agent:
  max_iterations: 5
  convergence_threshold: 0.95
  sample_size: 100

data:
  input_path: "data/raw/input.csv"
  output_path: "data/cleaned/output.csv"

logging:
  level: "INFO"
"""

    # 创建目录
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # 创建配置文件
    with open("config.yaml", "w") as f:
        f.write(config_content)

    # 创建示例数据目录
    Path("data/raw/.gitkeep").touch()
    Path("data/cleaned/.gitkeep").touch()

    click.echo("Initialized data laundry project!")
    click.echo("Created:")
    click.echo("  - config.yaml")
    click.echo("  - data/raw/")
    click.echo("  - data/cleaned/")
    click.echo("  - logs/")
    click.echo("\nSet your API key:")
    click.echo("  export OPENAI_API_KEY='your-key-here'")
    click.echo("  export ANTHROPIC_API_KEY='your-key-here'")


if __name__ == "__main__":
    cli()
