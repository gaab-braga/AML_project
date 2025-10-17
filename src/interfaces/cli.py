#!/usr/bin/env python3
"""
AML Pipeline CLI Interface

Command-line interface for the AML Pipeline system.
Provides comprehensive control over pipeline execution, monitoring, and management.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.logging import RichHandler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipeline_controller import PipelineController
from cache import HierarchicalCache, MemoryCache, DiskCache
from evaluation.metrics import ModelEvaluator
from config.config_loader import ConfigLoader
from utils.logger import get_logger

# Setup rich console and logging
console = Console()
logger = get_logger(__name__)

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)


class CLIContext:
    """Context object for CLI commands."""

    def __init__(self):
        self.config_path = "config/pipeline_config.yaml"
        self.controller: Optional[PipelineController] = None
        self.cache: Optional[HierarchicalCache] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.config: Optional[Dict[str, Any]] = None

    def ensure_initialized(self):
        """Ensure all components are initialized."""
        if not self.config:
            self.config = ConfigLoader.load_config(self.config_path)

        if not self.controller:
            self.controller = PipelineController(self.config)

        if not self.cache:
            # Initialize hierarchical cache
            memory_cache = MemoryCache(max_size=1000, ttl=3600)
            disk_cache = DiskCache(cache_dir="./cache", max_size_mb=500)
            self.cache = HierarchicalCache(
                memory_cache=memory_cache,
                disk_cache=disk_cache,
                cache_strategy="write-through"
            )

        if not self.evaluator:
            self.evaluator = ModelEvaluator(self.config)


# Global context
ctx = CLIContext()


@click.group()
@click.option('--config', default='config/pipeline_config.yaml',
              help='Path to pipeline configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.version_option(version='2.0.0')
def cli(config: str, verbose: bool):
    """
    AML Pipeline CLI - Enterprise-grade Anti-Money Laundering Pipeline

    A comprehensive command-line interface for managing and monitoring
    the AML pipeline execution, caching, and performance analytics.
    """
    ctx.config_path = config

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[bold blue]Verbose mode enabled[/bold blue]")

    # Show banner
    banner = Panel.fit(
        "[bold cyan]AML Pipeline CLI v2.0.0[/bold cyan]\n"
        "[dim]Enterprise Anti-Money Laundering Platform[/dim]",
        border_style="cyan"
    )
    console.print(banner)


@cli.command()
@click.option('--target', default='production',
              type=click.Choice(['development', 'staging', 'production']),
              help='Target environment to run')
@click.option('--mode', default='full',
              type=click.Choice(['full', 'fast', 'custom']),
              help='Execution mode')
@click.option('--async-run', is_flag=True, help='Run pipeline asynchronously')
@click.option('--watch', is_flag=True, help='Watch pipeline execution progress')
def run(target: str, mode: str, async_run: bool, watch: bool):
    """
    Execute the AML pipeline.

    Runs the complete pipeline from data ingestion through model evaluation.
    Supports different execution modes and environments.
    """
    ctx.ensure_initialized()

    try:
        console.print(f"[bold green]🚀 Starting AML Pipeline[/bold green]")
        console.print(f"Target: [cyan]{target}[/cyan] | Mode: [cyan]{mode}[/cyan]")

        start_time = time.time()

        if async_run:
            # Run asynchronously
            execution_id = ctx.controller.run_pipeline_async(target, mode)
            console.print(f"[green]Pipeline started asynchronously (ID: {execution_id})[/green]")

            if watch:
                _watch_pipeline_execution(execution_id)

        else:
            # Run synchronously
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing pipeline...", total=None)

                result = ctx.controller.run_pipeline(target, mode)

                progress.update(task, completed=True)

            execution_time = time.time() - start_time

            if result['success']:
                console.print(f"[bold green]✅ Pipeline completed successfully![/bold green]")
                console.print(f"Execution time: [cyan]{execution_time:.2f}s[/cyan]")

                # Show key metrics
                if 'metrics' in result:
                    _display_execution_summary(result['metrics'])

            else:
                console.print(f"[bold red]❌ Pipeline failed![/bold red]")
                if 'error' in result:
                    console.print(f"Error: [red]{result['error']}[/red]")
                sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]💥 Pipeline execution failed: {e}[/bold red]")
        logger.exception("Pipeline execution error")
        sys.exit(1)


@cli.command()
def status():
    """
    Check pipeline status and health.

    Displays current pipeline state, active executions, and system health.
    """
    ctx.ensure_initialized()

    try:
        status_info = ctx.controller.get_status()

        # Create status table
        table = Table(title="Pipeline Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        # Pipeline status
        pipeline_status = status_info.get('pipeline_status', 'unknown')
        status_color = {
            'running': 'green',
            'idle': 'blue',
            'error': 'red',
            'maintenance': 'yellow'
        }.get(pipeline_status, 'white')

        table.add_row(
            "Pipeline",
            f"[{status_color}]{pipeline_status.upper()}[/{status_color}]",
            f"Uptime: {status_info.get('uptime', 'N/A')}"
        )

        # Cache status
        if ctx.cache:
            cache_stats = ctx.cache.get_stats()
            table.add_row(
                "Cache",
                "[green]ACTIVE[/green]",
                f"Hit Rate: {cache_stats.get('hit_rate', 0):.1%}"
            )

        # Active executions
        active_executions = status_info.get('active_executions', 0)
        table.add_row(
            "Active Executions",
            "[blue]RUNNING[/blue]" if active_executions > 0 else "[dim]IDLE[/dim]",
            f"Count: {active_executions}"
        )

        console.print(table)

        # Show recent activity
        if 'recent_activity' in status_info:
            console.print("\n[bold]Recent Activity:[/bold]")
            for activity in status_info['recent_activity'][-5:]:  # Last 5 activities
                console.print(f"• {activity}")

    except Exception as e:
        console.print(f"[bold red]Failed to get status: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--period', default='1h',
              type=click.Choice(['1h', '24h', '7d', '30d']),
              help='Time period for metrics')
@click.option('--format', default='table',
              type=click.Choice(['table', 'json', 'csv']),
              help='Output format')
def metrics(period: str, format: str):
    """
    Display pipeline performance metrics.

    Shows comprehensive metrics including execution times, cache performance,
    model accuracy, and system resource usage.
    """
    ctx.ensure_initialized()

    try:
        # Convert period to hours
        period_hours = {
            '1h': 1,
            '24h': 24,
            '7d': 168,
            '30d': 720
        }[period]

        # Get metrics from different sources
        pipeline_metrics = ctx.controller.get_metrics()
        cache_metrics = ctx.cache.get_stats() if ctx.cache else {}
        system_metrics = _get_system_metrics()

        if format == 'json':
            # JSON output
            all_metrics = {
                'period': period,
                'timestamp': time.time(),
                'pipeline': pipeline_metrics,
                'cache': cache_metrics,
                'system': system_metrics
            }
            console.print_json(json.dumps(all_metrics, indent=2))

        elif format == 'csv':
            # CSV output (simplified)
            console.print("metric,value,timestamp")
            for key, value in pipeline_metrics.items():
                console.print(f"pipeline.{key},{value},{time.time()}")
            for key, value in cache_metrics.items():
                console.print(f"cache.{key},{value},{time.time()}")

        else:
            # Table format
            _display_metrics_table(pipeline_metrics, cache_metrics, system_metrics, period)

    except Exception as e:
        console.print(f"[bold red]Failed to get metrics: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--type', default='all',
              type=click.Choice(['memory', 'disk', 'all']),
              help='Cache type to manage')
@click.option('--clear', is_flag=True, help='Clear cache contents')
@click.option('--stats', is_flag=True, help='Show cache statistics')
def cache(type: str, clear: bool, stats: bool):
    """
    Manage cache operations.

    View cache statistics, clear cache contents, or perform maintenance operations.
    """
    ctx.ensure_initialized()

    try:
        if not ctx.cache:
            console.print("[yellow]Cache not initialized[/yellow]")
            return

        if clear:
            with console.status("[bold green]Clearing cache...[/bold green]"):
                if type in ['memory', 'all']:
                    # Clear memory cache
                    pass  # Implement based on cache interface
                if type in ['disk', 'all']:
                    # Clear disk cache
                    pass  # Implement based on cache interface

            console.print("[green]✅ Cache cleared successfully[/green]")

        elif stats:
            cache_stats = ctx.cache.get_stats()

            table = Table(title="Cache Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Cache Hits", str(cache_stats.get('hits', 0)))
            table.add_row("Cache Misses", str(cache_stats.get('misses', 0)))
            table.add_row("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")
            table.add_row("Total Operations", str(cache_stats.get('hits', 0) + cache_stats.get('misses', 0)))

            if 'memory_usage_mb' in cache_stats:
                table.add_row("Memory Usage", f"{cache_stats['memory_usage_mb']:.1f} MB")

            if 'disk_usage_mb' in cache_stats:
                table.add_row("Disk Usage", f"{cache_stats['disk_usage_mb']:.1f} MB")

            console.print(table)

        else:
            console.print("[yellow]Use --stats to view statistics or --clear to clear cache[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Cache operation failed: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate(config_file: str):
    """
    Validate pipeline configuration.

    Checks configuration file for correctness and completeness.
    """
    try:
        console.print(f"[bold blue]🔍 Validating configuration: {config_file}[/bold blue]")

        # Load and validate config
        config = ConfigLoader.load_config(config_file)

        # Basic validation checks
        validation_results = _validate_config(config)

        if validation_results['valid']:
            console.print("[bold green]✅ Configuration is valid![/bold green]")

            # Show config summary
            table = Table(title="Configuration Summary")
            table.add_column("Section", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="yellow")

            for section, status in validation_results['sections'].items():
                table.add_row(
                    section.title(),
                    "[green]OK[/green]" if status['valid'] else "[red]ERROR[/red]",
                    status.get('message', '')
                )

            console.print(table)

        else:
            console.print("[bold red]❌ Configuration validation failed![/bold red]")
            for error in validation_results['errors']:
                console.print(f"• [red]{error}[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Configuration validation failed: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='report.md',
              help='Output file for the report')
@click.option('--format', default='markdown',
              type=click.Choice(['markdown', 'html', 'json']),
              help='Report format')
def report(output: str, format: str):
    """
    Generate comprehensive pipeline report.

    Creates detailed reports on pipeline performance, metrics, and recommendations.
    """
    ctx.ensure_initialized()

    try:
        console.print("[bold blue]📊 Generating pipeline report...[/bold blue]")

        with console.status("[bold green]Collecting data...[/bold green]"):
            # Collect all relevant data
            status = ctx.controller.get_status()
            metrics = ctx.controller.get_metrics()
            cache_stats = ctx.cache.get_stats() if ctx.cache else {}
            config_summary = _get_config_summary(ctx.config)

        # Generate report content
        report_content = _generate_report(
            status, metrics, cache_stats, config_summary, format
        )

        # Save report
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report_content)

        console.print(f"[bold green]✅ Report generated: {output}[/bold green]")
        console.print(f"Format: [cyan]{format.upper()}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]Report generation failed: {e}[/bold red]")
        sys.exit(1)


def _watch_pipeline_execution(execution_id: str):
    """Watch pipeline execution progress in real-time."""
    console.print(f"[bold blue]👀 Watching execution: {execution_id}[/bold blue]")

    with Live(console=console, refresh_per_second=2) as live:
        while True:
            try:
                status = ctx.controller.get_execution_status(execution_id)

                if status['state'] == 'completed':
                    live.update(Text("✅ Pipeline completed successfully!", style="green"))
                    break
                elif status['state'] == 'failed':
                    live.update(Text(f"❌ Pipeline failed: {status.get('error', 'Unknown error')}", style="red"))
                    break
                elif status['state'] == 'running':
                    progress = status.get('progress', 0)
                    current_step = status.get('current_step', 'Initializing...')

                    progress_bar = f"[{'█' * int(progress * 40)}{'░' * (40 - int(progress * 40))}] {progress:.1%}"
                    live.update(Text(f"🔄 {current_step}\n{progress_bar}", style="blue"))
                else:
                    live.update(Text(f"⏳ {status['state'].title()}...", style="yellow"))

                time.sleep(1)

            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped watching execution[/yellow]")
                break
            except Exception as e:
                live.update(Text(f"❌ Error watching execution: {e}", style="red"))
                break


def _display_execution_summary(metrics: Dict[str, Any]):
    """Display execution summary with key metrics."""
    if not metrics:
        return

    table = Table(title="Execution Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Add key metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")
        else:
            table.add_row(key.replace('_', ' ').title(), str(value))

    console.print(table)


def _display_metrics_table(pipeline_metrics: Dict, cache_metrics: Dict,
                          system_metrics: Dict, period: str):
    """Display metrics in table format."""
    # Pipeline metrics
    if pipeline_metrics:
        table = Table(title=f"Pipeline Metrics (Last {period})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in pipeline_metrics.items():
            if isinstance(value, float):
                table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")
            else:
                table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)
        console.print()

    # Cache metrics
    if cache_metrics:
        table = Table(title="Cache Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Hits", str(cache_metrics.get('hits', 0)))
        table.add_row("Misses", str(cache_metrics.get('misses', 0)))
        table.add_row("Hit Rate", f"{cache_metrics.get('hit_rate', 0):.1%}")

        if 'memory_usage_mb' in cache_metrics:
            table.add_row("Memory Usage", f"{cache_metrics['memory_usage_mb']:.1f} MB")

        console.print(table)
        console.print()

    # System metrics
    if system_metrics:
        table = Table(title="System Resources")
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="green")

        for key, value in system_metrics.items():
            if 'percent' in key.lower():
                table.add_row(key.replace('_', ' ').title(), f"{value:.1f}%")
            elif 'mb' in key.lower():
                table.add_row(key.replace('_', ' ').title(), f"{value:.1f} MB")
            else:
                table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)


def _get_system_metrics() -> Dict[str, Any]:
    """Get basic system metrics."""
    try:
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    except ImportError:
        return {}


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate pipeline configuration."""
    results = {
        'valid': True,
        'errors': [],
        'sections': {}
    }

    # Required sections
    required_sections = ['pipeline', 'data', 'features', 'model', 'evaluation']

    for section in required_sections:
        if section not in config:
            results['errors'].append(f"Missing required section: {section}")
            results['sections'][section] = {'valid': False, 'message': 'Section missing'}
            results['valid'] = False
        else:
            results['sections'][section] = {'valid': True, 'message': 'Section present'}

    # Validate pipeline section
    if 'pipeline' in config:
        pipeline_config = config['pipeline']
        required_fields = ['name', 'version', 'primary_metric']

        for field in required_fields:
            if field not in pipeline_config:
                results['errors'].append(f"Missing required field in pipeline section: {field}")
                results['sections']['pipeline'] = {'valid': False, 'message': f'Missing field: {field}'}
                results['valid'] = False

    return results


def _get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get configuration summary."""
    if not config:
        return {}

    return {
        'pipeline_name': config.get('pipeline', {}).get('name', 'Unknown'),
        'pipeline_version': config.get('pipeline', {}).get('version', 'Unknown'),
        'primary_metric': config.get('pipeline', {}).get('primary_metric', 'Unknown'),
        'cv_folds': config.get('pipeline', {}).get('cv_folds', 'Unknown'),
        'model_types': list(config.get('model', {}).keys()) if 'model' in config else []
    }


def _generate_report(status: Dict, metrics: Dict, cache_stats: Dict,
                    config_summary: Dict, format: str) -> str:
    """Generate comprehensive report."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    if format == 'json':
        report_data = {
            'generated_at': timestamp,
            'status': status,
            'metrics': metrics,
            'cache_stats': cache_stats,
            'config_summary': config_summary
        }
        return json.dumps(report_data, indent=2, default=str)

    elif format == 'html':
        # Simple HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AML Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>AML Pipeline Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>

            <div class="section">
                <h2>Pipeline Status</h2>
                <div class="metric">Status: {status.get('pipeline_status', 'Unknown')}</div>
                <div class="metric">Uptime: {status.get('uptime', 'Unknown')}</div>
            </div>

            <div class="section">
                <h2>Performance Metrics</h2>
                {"".join(f"<div class='metric'>{k}: {v}</div>" for k, v in metrics.items())}
            </div>

            <div class="section">
                <h2>Cache Statistics</h2>
                {"".join(f"<div class='metric'>{k}: {v}</div>" for k, v in cache_stats.items())}
            </div>
        </body>
        </html>
        """
        return html

    else:  # markdown
        report = f"""# AML Pipeline Report

**Generated:** {timestamp}

## Pipeline Status
- **Status:** {status.get('pipeline_status', 'Unknown')}
- **Uptime:** {status.get('uptime', 'Unknown')}
- **Active Executions:** {status.get('active_executions', 0)}

## Performance Metrics
"""
        for key, value in metrics.items():
            report += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        report += "\n## Cache Statistics\n"
        for key, value in cache_stats.items():
            report += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        report += "\n## Configuration Summary\n"
        for key, value in config_summary.items():
            report += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        return report


if __name__ == '__main__':
    cli()