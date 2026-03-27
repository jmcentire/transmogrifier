# Transmogrifier

Register-aware prompt translation. Detects linguistic register and normalizes it to maximize LLM output quality.

## Why

LLM output quality varies by input register. Empirical testing across production models:

| Model | Spread | Best Register | Worst Register |
|-------|--------|--------------|----------------|
| Claude Opus 4 | 18.8pp | direct/technical | casual (75%) |
| Gemini 2.5 Flash | 56.2pp | direct/technical | narrative (0%) |
| Claude Haiku 4.5 | 6.2pp | direct | casual |
| GPT-4o Mini | 0pp | (invariant) | (invariant) |

Casual register costs Claude Opus ~19% of correct answers. Gemini loses over half. Transmogrifier fixes this transparently.

## Install

```bash
pip install transmogrifier
```

With optional backends for Level 3 LLM-based translation:

```bash
pip install "transmogrifier[anthropic]"    # Claude backend
pip install "transmogrifier[openai]"       # OpenAI backend
pip install "transmogrifier[gemini]"       # Gemini backend
pip install "transmogrifier[validation]"   # Semantic equivalence checking
pip install "transmogrifier[all]"          # Everything
```

## Usage

### Python Library

```python
from transmogrifier.core import Transmogrifier

t = Transmogrifier()
result = t.translate("yo what's the deal with TCP", model="claude-opus-4")

result.output_text        # "TCP" (rewritten, filler stripped)
result.system_prompt      # Level 1 normalization instruction
result.detected_register  # Register.casual
result.target_register    # Register.direct
result.elapsed_ms         # ~2ms
result.skipped            # False (Opus is register-sensitive)
```

For register-invariant models, translation is skipped automatically:

```python
result = t.translate("yo what's TCP", model="gpt-4o-mini")
result.skipped      # True
result.skip_reason  # "invariant model (0.0pp spread)"
```

### CLI

```bash
# Detect register
transmogrify detect "yo so like, what's the deal with TCP"
# {"register": "casual", "confidence": 1.0}

# Translate with model-aware optimization
transmogrify translate "yo what's activation fingerprinting" --model claude-opus-4
# Detected:  casual
# Target:    direct
# Level:     rule_rewrite
# Output:    Activation fingerprinting

# List model profiles
transmogrify profile list
# claude-opus-4: spread=18.8pp best=direct
# gpt-4o-mini: spread=0.0pp best=direct (invariant)
# gemini-2-5-flash: spread=56.2pp best=direct
```

### MCP Server (Claude Code integration)

```bash
# Register as MCP server
claude mcp add --scope user --transport stdio transmogrifier -- transmog-mcp
```

Provides tools: `transmog_translate`, `transmog_detect`, `transmog_profiles`.

## How It Works

Three levels of intervention, applied based on model sensitivity:

**Level 1 -- System Prompt Injection (always, 0ms, no API calls)**
Prepends a register normalization instruction to the system prompt. Recovers 67-100% of register-induced accuracy loss.

**Level 2 -- Rule-Based Rewriting (when source != target register, <1ms, no API calls)**
Strips register-specific surface forms (casual filler, academic hedging, narrative framing) via regex templates. Preserves semantic content.

**Level 3 -- LLM Translation (optional, when spread >10pp and configured)**
Separate-context LLM call for heavy-lift register translation. Never same-context (catastrophic on Gemini).

## Architecture

- **Register Detector**: Heuristic classifier using surface-form markers. 5 registers: direct, casual, technical, academic, narrative.
- **Model Profile Cache**: Pre-seeded with empirical data from 4 models. JSON files at `~/.transmogrifier/profiles/`. Versioned and TTL'd.
- **Translation Router**: Selects Level 1/2/3 based on model sensitivity profile and configuration.
- **Semantic Validator** (optional): Embedding similarity check (>0.95 threshold) to catch semantic drift during translation.

## Key Invariants

1. **Separate context**: Level 3 translation is never in the same message as task execution
2. **No data persistence**: User prompts are never written to disk
3. **Fail-safe passthrough**: On any error, returns the original text unchanged
4. **Downstream override**: Callers can force a specific target register
5. **Zero-dependency L1+L2**: No API calls required for Level 1-2

## Integration

Designed as middleware for:
- **Constrain** -- before sending interview questions to LLM backend
- **Pact** -- before sending task specs and code generation prompts
- **Kindex** -- before embedding queries for retrieval (8.4% cosine similarity recovery)
- Any tool calling LLM APIs

## License

MIT
