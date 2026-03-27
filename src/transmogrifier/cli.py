"""Click CLI — thin wrapper over library. Phase 9 stub with detect/translate."""

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
