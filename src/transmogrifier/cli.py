"""Click CLI — thin wrapper over library."""

from __future__ import annotations

import json
import sys

try:
    import click
except ImportError:
    print("CLI requires click: pip install click", file=sys.stderr)
    sys.exit(1)

from .core import Register, Transmogrifier, TranslationConfig


@click.group()
def main():
    """Transmogrifier: register-aware prompt translation."""
    pass


@main.command()
@click.argument("text")
def detect(text: str):
    """Detect the register of input text."""
    t = Transmogrifier()
    detected, confidence = t._detector.detect(text)
    click.echo(json.dumps({"register": detected.value, "confidence": confidence}))


@main.command()
@click.argument("text")
def classify(text: str):
    """Classify the task type of input text."""
    t = Transmogrifier()
    task_type, confidence = t._task_classifier.classify(text)
    click.echo(json.dumps({"task_type": task_type.value, "confidence": confidence}))


@main.command()
@click.argument("text")
@click.option("--model", default="", help="Target model name for profile lookup")
@click.option("--register", "target", default=None, help="Force target register")
@click.option("--json-output", "as_json", is_flag=True, help="JSON output")
def translate(text: str, model: str, target: str | None, as_json: bool):
    """Translate text to optimal register for the target model."""
    config = TranslationConfig()
    if target:
        config.target_register = Register(target)

    t = Transmogrifier()
    result = t.translate(text, model=model, config=config)

    if as_json:
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(f"Detected:  {result.detected_register.value}")
        click.echo(f"Task:      {result.detected_task}")
        click.echo(f"Target:    {result.target_register.value}")
        click.echo(f"Level:     {result.level_applied.name}")
        click.echo(f"Skipped:   {result.skipped}")
        click.echo(f"Time:      {result.elapsed_ms:.2f}ms")
        if result.output_text != result.input_text:
            click.echo(f"Output:    {result.output_text}")
        if result.system_prompt:
            click.echo(f"SysPrompt: {result.system_prompt[:80]}...")


@main.group()
def profile():
    """Manage model register sensitivity profiles."""
    pass


@profile.command("list")
def profile_list():
    """List cached model profiles."""
    from .profiles import ProfileCache
    cache = ProfileCache()
    for p in cache.list_profiles():
        inv = " (invariant)" if p.is_invariant else ""
        click.echo(f"  {p.model_name}: spread={p.spread_pp:.1f}pp best={p.best_register}{inv}")
        for tp in p.by_task:
            click.echo(f"    {tp.task_type}: best={tp.best_register} spread={tp.spread_pp:.1f}pp")


@profile.command("show")
@click.argument("model_name")
def profile_show(model_name: str):
    """Show detailed profile for a model."""
    from .profiles import ProfileCache
    cache = ProfileCache()
    p = cache.get(model_name)
    if not p:
        click.echo(f"No profile found for '{model_name}'")
        return
    click.echo(f"Model:     {p.model_name} ({p.model_version})")
    click.echo(f"Provider:  {p.provider}")
    click.echo(f"Spread:    {p.spread_pp:.1f}pp")
    click.echo(f"Invariant: {p.is_invariant}")
    click.echo(f"Best:      {p.best_register}")
    click.echo(f"Worst:     {p.worst_register}")
    click.echo(f"Calibrated: {p.calibrated_at}")
    click.echo(f"\nAggregate:")
    for a in sorted(p.accuracies, key=lambda x: x.accuracy, reverse=True):
        bar = "#" * int(a.accuracy * 30)
        click.echo(f"  {a.register:<12s} {a.accuracy*100:5.1f}%  {bar}")
    if p.by_task:
        click.echo(f"\nPer task:")
        for tp in p.by_task:
            click.echo(f"  {tp.task_type}:")
            for a in sorted(tp.accuracies, key=lambda x: x.accuracy, reverse=True):
                marker = " *" if a.register == tp.best_register else ""
                click.echo(f"    {a.register:<12s} {a.accuracy*100:5.1f}%{marker}")


@profile.command("calibrate")
@click.argument("model_name")
@click.option("--provider", default="anthropic", help="Backend provider")
@click.option("--model-id", default=None, help="API model ID (if different from profile name)")
@click.option("--version", default="", help="Model version string")
@click.option("--quick", is_flag=True, help="Run with reduced task set (10 tasks)")
def profile_calibrate(model_name: str, provider: str, model_id: str | None, version: str, quick: bool):
    """Run calibration benchmark for a model."""
    from .backends import create_backend
    from .calibrate import BENCHMARK_TASKS, CalibrationRunner
    from .profiles import ProfileCache

    backend = create_backend(provider, model=model_id)
    cache = ProfileCache()
    runner = CalibrationRunner(backend, cache)

    tasks = BENCHMARK_TASKS[:10] if quick else BENCHMARK_TASKS

    click.echo(f"Calibrating {model_name} ({provider})...")
    click.echo(f"Tasks: {len(tasks)} across {len(set(t['category'] for t in tasks))} categories")
    click.echo(f"Registers: 5 x {len(tasks)} = {5 * len(tasks)} API calls")
    click.echo()

    profile = runner.run(
        model_name=model_name,
        model_version=version,
        provider=provider,
        tasks=tasks,
        verbose=True,
    )

    click.echo(f"\nDone. Spread: {profile.spread_pp:.1f}pp, Best: {profile.best_register}")
