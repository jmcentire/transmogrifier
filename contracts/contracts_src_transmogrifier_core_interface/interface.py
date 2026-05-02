# === Transmogrifier Core Interface (contracts_src_transmogrifier_core_interface) v1 ===
#  Dependencies: time, uuid, pydantic, src.transmogrifier.detector, src.transmogrifier.profiles, src.transmogrifier.rules, src.transmogrifier.system_prompts, src.transmogrifier.task_classifier
# Core orchestrator for register-aware prompt translation. Detects input register, determines optimal target register (optionally task-aware), and rewrites prompts using rule-based transformations with system prompt injection.

# Module invariants:
#   - TranslationLevel.system_prompt (1) < TranslationLevel.rule_rewrite (2) < TranslationLevel.llm_translate (3)
#   - task_spread threshold for skipping is hardcoded to 2.0
#   - Default target register is Register.direct when no config or profile available
#   - System prompt is always generated regardless of skip status
#   - trace_id is generated via uuid.uuid4().hex[:12] for each TranslationResult
#   - elapsed_ms is always measured and populated in result

class Register(Enum):
    """Available linguistic registers for prompt translation"""
    direct = "direct"
    casual = "casual"
    technical = "technical"
    academic = "academic"
    narrative = "narrative"

class TranslationLevel(Enum):
    """Transformation level applied during translation"""
    system_prompt = "system_prompt"
    rule_rewrite = "rule_rewrite"
    llm_translate = "llm_translate"

class TranslationResult:
    """Complete result of a translation operation with metadata"""
    input_text: str                          # required, Original input text
    output_text: str                         # required, Translated/rewritten output text
    detected_register: Register              # required, Register detected in input
    target_register: Register                # required, Target register used for translation
    detected_task: str | None = None         # optional, Task type classified from input
    level_applied: TranslationLevel          # required, Highest translation level applied
    system_prompt: str | None = None         # optional, Generated system prompt for LLM injection
    semantic_similarity: float | None = None # optional, Semantic similarity score (unused in current code)
    skipped: bool = False                    # optional, Whether translation was skipped
    skip_reason: str | None = None           # optional, Reason for skipping translation
    elapsed_ms: float = 0.0                  # optional, Translation duration in milliseconds
    trace_id: str | None = None              # optional, 12-character hex trace ID generated via uuid4

class TranslationConfig:
    """Configuration parameters for translation behavior"""
    target_register: Register | None = None  # optional, Force specific target register
    max_level: TranslationLevel = TranslationLevel.rule_rewrite # optional, Maximum translation level to apply
    semantic_threshold: float = 0.95         # optional, Semantic similarity threshold (unused in current code)
    spread_threshold_pp: float = 10.0        # optional, Percentage point spread threshold for invariance
    passthrough_on_failure: bool = True      # optional, Return input text on failure (unused in current code)
    task_aware: bool = True                  # optional, Use per-task optimal register when available

def __init__(
    self: Transmogrifier,
    profile_cache: ProfileCache | None = None,
    config: TranslationConfig | None = None,
) -> None:
    """
    Initialize Transmogrifier with detector, task classifier, profile cache, rule engine, and configuration. Creates default instances if not provided.

    Postconditions:
      - self._detector is RegisterDetector instance
      - self._task_classifier is TaskClassifier instance
      - self._profile_cache is ProfileCache instance (provided or default)
      - self._rule_engine is RuleEngine instance
      - self._config is TranslationConfig instance (provided or default)

    Side effects: Creates RegisterDetector instance, Creates TaskClassifier instance, Creates RuleEngine instance, Creates default ProfileCache if not provided, Creates default TranslationConfig if not provided
    Idempotent: no
    """
    ...

def translate(
    self: Transmogrifier,
    text: str,
    model: str | None = None,
    config: TranslationConfig | None = None,
) -> TranslationResult:
    """
    Translate input text to optimal register using rule-based rewriting and system prompt generation. Detects input register, classifies task type, determines target register (from config, model profile, or default to direct), generates system prompt, and applies rule rewriting if registers differ. Skips translation for invariant models with low task-specific spread. Zero API calls.

    Preconditions:
      - self._detector, self._task_classifier, self._profile_cache, self._rule_engine, self._config are initialized

    Postconditions:
      - Returns TranslationResult with input_text == text
      - result.detected_register is the detected register from input
      - result.target_register is determined from config, profile, or defaults to Register.direct
      - result.system_prompt is generated by get_system_prompt(detected, target)
      - result.output_text == text if skipped or detected == target
      - result.output_text is rewritten by RuleEngine if detected != target
      - result.level_applied is TranslationLevel.system_prompt if detected == target
      - result.level_applied is TranslationLevel.rule_rewrite if detected != target
      - result.skipped == True if profile.is_invariant and task_spread < 2.0
      - result.elapsed_ms is measured execution time in milliseconds

    Errors:
      - ValueError_on_invalid_register_string (ValueError): target from profile.best_register or profile.best_register_for_task is a string not in Register enum
          source: Register(target) conversion

    Side effects: Calls self._detector.detect(text), Calls self._task_classifier.classify(text), Calls self._profile_cache.get(model) if model is non-empty, Calls get_system_prompt(detected, target), Calls self._rule_engine.rewrite(text, detected, target) if detected != target, Measures elapsed time using time.perf_counter()
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['Register', 'TranslationLevel', 'TranslationResult', 'TranslationConfig', 'translate']
