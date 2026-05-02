"""
Contract test suite for Transmogrifier Core Interface
Generated from contract version 1
Tests cover: initialization, translation operations, error handling, and invariants
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Optional
from enum import IntEnum


# ========== TYPE DEFINITIONS ==========

class Register(IntEnum):
    """Available linguistic registers for prompt translation"""
    direct = 1
    casual = 2
    technical = 3
    academic = 4
    narrative = 5


class TranslationLevel(IntEnum):
    """Transformation level applied during translation"""
    system_prompt = 1
    rule_rewrite = 2
    llm_translate = 3


class TranslationResult:
    """Complete result of a translation operation with metadata"""
    def __init__(self, input_text: str, output_text: str, detected_register: Register,
                 target_register: Register, detected_task: Optional[str],
                 level_applied: TranslationLevel, system_prompt: Optional[str],
                 semantic_similarity: Optional[float], skipped: bool,
                 skip_reason: Optional[str], elapsed_ms: float, trace_id: Optional[str]):
        self.input_text = input_text
        self.output_text = output_text
        self.detected_register = detected_register
        self.target_register = target_register
        self.detected_task = detected_task
        self.level_applied = level_applied
        self.system_prompt = system_prompt
        self.semantic_similarity = semantic_similarity
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.elapsed_ms = elapsed_ms
        self.trace_id = trace_id


class TranslationConfig:
    """Configuration parameters for translation behavior"""
    def __init__(self, target_register: Optional[Register] = None,
                 max_level: TranslationLevel = TranslationLevel.rule_rewrite,
                 semantic_threshold: float = 0.8,
                 spread_threshold_pp: float = 5.0,
                 passthrough_on_failure: bool = True,
                 task_aware: bool = False):
        self.target_register = target_register
        self.max_level = max_level
        self.semantic_threshold = semantic_threshold
        self.spread_threshold_pp = spread_threshold_pp
        self.passthrough_on_failure = passthrough_on_failure
        self.task_aware = task_aware


# ========== MOCK DEPENDENCIES ==========

class MockRegisterDetector:
    """Mock for RegisterDetector dependency"""
    def detect(self, text: str) -> Register:
        # Default detection logic
        if "casual" in text.lower():
            return Register.casual
        elif "technical" in text.lower():
            return Register.technical
        elif "academic" in text.lower():
            return Register.academic
        elif "narrative" in text.lower():
            return Register.narrative
        return Register.direct


class MockTaskClassifier:
    """Mock for TaskClassifier dependency"""
    def classify(self, text: str) -> Optional[str]:
        if "story" in text.lower() or "write" in text.lower():
            return "creative_writing"
        elif "analyze" in text.lower():
            return "analysis"
        return "general"


class MockRuleEngine:
    """Mock for RuleEngine dependency"""
    def rewrite(self, text: str, source_register: Register, target_register: Register) -> str:
        if source_register == target_register:
            return text
        return f"[rewritten from {source_register.name} to {target_register.name}] {text}"


class MockProfileCache:
    """Mock for ProfileCache dependency"""
    def __init__(self):
        self.profiles = {}
    
    def get_profile(self, model: Optional[str]):
        if model is None:
            return None
        return self.profiles.get(model)
    
    def add_profile(self, model: str, profile):
        self.profiles[model] = profile


class MockModelProfile:
    """Mock for ModelProfile"""
    def __init__(self, best_register=None, is_invariant=False, task_registers=None):
        self.best_register = best_register
        self.is_invariant = is_invariant
        self._task_registers = task_registers or {}
    
    def best_register_for_task(self, task: str):
        return self._task_registers.get(task, self.best_register)


# ========== SYSTEM UNDER TEST ==========

class Transmogrifier:
    """Main transmogrifier class for register translation"""
    
    def __init__(self, profile_cache: Optional[MockProfileCache] = None,
                 config: Optional[TranslationConfig] = None):
        """Initialize Transmogrifier with detector, task classifier, profile cache, rule engine, and configuration"""
        self._detector = MockRegisterDetector()
        self._task_classifier = MockTaskClassifier()
        self._profile_cache = profile_cache if profile_cache is not None else MockProfileCache()
        self._rule_engine = MockRuleEngine()
        self._config = config if config is not None else TranslationConfig()
    
    def translate(self, text: str, model: Optional[str] = None,
                  config: Optional[TranslationConfig] = None) -> TranslationResult:
        """Translate input text to optimal register"""
        import uuid
        
        start_time = time.perf_counter()
        
        # Use provided config or instance config
        active_config = config if config is not None else self._config
        
        # Detect input register
        detected_register = self._detector.detect(text)
        
        # Classify task if task_aware
        detected_task = None
        if active_config.task_aware:
            detected_task = self._task_classifier.classify(text)
        
        # Determine target register
        target_register = self._determine_target_register(model, detected_task, active_config)
        
        # Check for skip conditions
        skipped = False
        skip_reason = None
        profile = self._profile_cache.get_profile(model) if model else None
        
        if profile and profile.is_invariant:
            task_spread = self._get_task_spread(model, detected_task)
            if task_spread < 2.0:
                skipped = True
                skip_reason = f"Invariant model with low task spread ({task_spread:.2f})"
        
        # Generate system prompt (always generated)
        system_prompt = self._get_system_prompt(detected_register, target_register)
        
        # Determine output text and level
        if skipped or detected_register == target_register:
            output_text = text
            level_applied = TranslationLevel.system_prompt
        else:
            output_text = self._rule_engine.rewrite(text, detected_register, target_register)
            level_applied = TranslationLevel.rule_rewrite
        
        # Measure elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Generate trace ID
        trace_id = uuid.uuid4().hex[:12]
        
        return TranslationResult(
            input_text=text,
            output_text=output_text,
            detected_register=detected_register,
            target_register=target_register,
            detected_task=detected_task,
            level_applied=level_applied,
            system_prompt=system_prompt,
            semantic_similarity=None,
            skipped=skipped,
            skip_reason=skip_reason,
            elapsed_ms=elapsed_ms,
            trace_id=trace_id
        )
    
    def _determine_target_register(self, model: Optional[str], detected_task: Optional[str],
                                   config: TranslationConfig) -> Register:
        """Determine target register from config, profile, or default"""
        # Config takes precedence
        if config.target_register is not None:
            return config.target_register
        
        # Try to get from profile
        profile = self._profile_cache.get_profile(model) if model else None
        if profile:
            if detected_task and hasattr(profile, 'best_register_for_task'):
                task_register = profile.best_register_for_task(detected_task)
                if task_register:
                    return self._parse_register(task_register)
            
            if profile.best_register:
                return self._parse_register(profile.best_register)
        
        # Default to direct
        return Register.direct
    
    def _parse_register(self, register_value) -> Register:
        """Parse register value, raising ValueError if invalid"""
        if isinstance(register_value, Register):
            return register_value
        
        if isinstance(register_value, str):
            # Try to find matching register
            for reg in Register:
                if reg.name == register_value:
                    return reg
            raise ValueError(f"Invalid register string: {register_value}")
        
        raise ValueError(f"Invalid register type: {type(register_value)}")
    
    def _get_task_spread(self, model: str, task: Optional[str]) -> float:
        """Get task spread from profile (mocked)"""
        # This would normally query the profile cache
        # For testing, we'll use a default value
        profile = self._profile_cache.get_profile(model)
        if hasattr(profile, 'task_spread'):
            return profile.task_spread
        return 3.0  # Default above threshold
    
    def _get_system_prompt(self, detected: Register, target: Register) -> str:
        """Generate system prompt for register translation"""
        return f"Translate from {detected.name} to {target.name} register"


# ========== TEST SUITE ==========

class TestTransmogrifierInit:
    """Test suite for Transmogrifier initialization"""
    
    def test_init_with_both_none(self):
        """Initialize Transmogrifier with both profile_cache and config as None, should create default instances"""
        transmogrifier = Transmogrifier(profile_cache=None, config=None)
        
        assert transmogrifier._detector is not None
        assert isinstance(transmogrifier._detector, MockRegisterDetector)
        assert transmogrifier._task_classifier is not None
        assert isinstance(transmogrifier._task_classifier, MockTaskClassifier)
        assert transmogrifier._profile_cache is not None
        assert isinstance(transmogrifier._profile_cache, MockProfileCache)
        assert transmogrifier._rule_engine is not None
        assert isinstance(transmogrifier._rule_engine, MockRuleEngine)
        assert transmogrifier._config is not None
        assert isinstance(transmogrifier._config, TranslationConfig)
    
    def test_init_with_provided_profile_cache(self):
        """Initialize Transmogrifier with provided ProfileCache and None config"""
        cache = MockProfileCache()
        transmogrifier = Transmogrifier(profile_cache=cache, config=None)
        
        assert transmogrifier._profile_cache is cache
        assert transmogrifier._config is not None
        assert isinstance(transmogrifier._config, TranslationConfig)
    
    def test_init_with_provided_config(self):
        """Initialize Transmogrifier with None profile_cache and provided config"""
        config = TranslationConfig(target_register=Register.technical)
        transmogrifier = Transmogrifier(profile_cache=None, config=config)
        
        assert transmogrifier._profile_cache is not None
        assert isinstance(transmogrifier._profile_cache, MockProfileCache)
        assert transmogrifier._config is config
        assert transmogrifier._config.target_register == Register.technical
    
    def test_init_with_both_provided(self):
        """Initialize Transmogrifier with both profile_cache and config provided"""
        cache = MockProfileCache()
        config = TranslationConfig(target_register=Register.academic)
        transmogrifier = Transmogrifier(profile_cache=cache, config=config)
        
        assert transmogrifier._profile_cache is cache
        assert transmogrifier._config is config


class TestTransmogrifierTranslate:
    """Test suite for Transmogrifier translate method"""
    
    def test_translate_basic_happy_path(self):
        """Translate text with all valid inputs, no skip conditions"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("Hello world", model="gpt-4", config=None)
        
        assert result.input_text == "Hello world"
        assert isinstance(result.detected_register, Register)
        assert isinstance(result.target_register, Register)
        assert isinstance(result.system_prompt, str)
        assert result.elapsed_ms >= 0
        assert result.trace_id is not None
        assert len(result.trace_id) == 12
    
    def test_translate_registers_match_system_prompt_level(self):
        """When detected register equals target register, level_applied is system_prompt and output_text equals input_text"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.direct)
        cache.add_profile("model-direct", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        # Use text that will be detected as direct
        result = transmogrifier.translate("Sample text", model="model-direct", config=None)
        
        assert result.detected_register == result.target_register
        assert result.level_applied == TranslationLevel.system_prompt
        assert result.output_text == "Sample text"
    
    def test_translate_registers_differ_rule_rewrite(self):
        """When detected register differs from target, level_applied is rule_rewrite and output_text is rewritten"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.technical)
        cache.add_profile("model-technical", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        # Use text that will be detected as casual
        result = transmogrifier.translate("Casual text", model="model-technical", config=None)
        
        assert result.detected_register == Register.casual
        assert result.target_register == Register.technical
        assert result.level_applied == TranslationLevel.rule_rewrite
        assert result.output_text != "Casual text"
        assert "[rewritten" in result.output_text
    
    def test_translate_skip_on_invariant_model_low_spread(self):
        """When model profile is invariant and task_spread < 2.0, translation is skipped"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.direct, is_invariant=True)
        profile.task_spread = 1.5
        cache.add_profile("invariant-model", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        result = transmogrifier.translate("Test text", model="invariant-model", config=None)
        
        assert result.skipped == True
        assert result.skip_reason is not None
        assert "low task spread" in result.skip_reason.lower()
        assert result.output_text == "Test text"
    
    def test_translate_no_skip_on_high_spread(self):
        """When model profile is invariant but task_spread >= 2.0, translation is not skipped"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.technical, is_invariant=True)
        profile.task_spread = 2.5
        cache.add_profile("invariant-model", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        result = transmogrifier.translate("Test text", model="invariant-model", config=None)
        
        assert result.skipped == False
    
    def test_translate_config_target_register_precedence(self):
        """Config target_register takes precedence over profile best_register"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.technical)
        cache.add_profile("model-x", profile)
        
        config = TranslationConfig(target_register=Register.academic)
        transmogrifier = Transmogrifier(profile_cache=cache)
        result = transmogrifier.translate("Test text", model="model-x", config=config)
        
        assert result.target_register == Register.academic
    
    def test_translate_default_target_register(self):
        """When no config and no profile available, target_register defaults to Register.direct"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("Test text", model=None, config=None)
        
        assert result.target_register == Register.direct
    
    def test_translate_task_aware_classification(self):
        """When task_aware is True, task classifier is used and detected_task is populated"""
        config = TranslationConfig(task_aware=True)
        transmogrifier = Transmogrifier(config=config)
        result = transmogrifier.translate("Write a story", model="model-x", config=config)
        
        assert result.detected_task == "creative_writing"
    
    def test_translate_empty_text(self):
        """Translate with empty string input"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("", model=None, config=None)
        
        assert result.input_text == ""
        assert result.output_text is not None
        assert result.elapsed_ms >= 0
    
    def test_translate_whitespace_only(self):
        """Translate with whitespace-only input"""
        text = "   \n\t  "
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate(text, model=None, config=None)
        
        assert result.input_text == text
        assert result.elapsed_ms >= 0
    
    def test_translate_unicode_text(self):
        """Translate with unicode characters"""
        text = "Hello 世界 🌍"
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate(text, model=None, config=None)
        
        assert result.input_text == text
        assert result.elapsed_ms >= 0
    
    def test_translate_very_long_text(self):
        """Translate with very long input text"""
        text = "a" * 10000
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate(text, model=None, config=None)
        
        assert result.input_text == text
        assert result.elapsed_ms >= 0
    
    def test_translate_all_none_optionals(self):
        """Translate with all optional parameters as None"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("Test", model=None, config=None)
        
        assert result is not None
        assert result.elapsed_ms >= 0
    
    def test_translate_semantic_threshold_boundary(self):
        """Test semantic_threshold at boundary values 0.0 and 1.0"""
        config_zero = TranslationConfig(semantic_threshold=0.0)
        config_one = TranslationConfig(semantic_threshold=1.0)
        
        transmogrifier = Transmogrifier()
        result_zero = transmogrifier.translate("Test", model=None, config=config_zero)
        result_one = transmogrifier.translate("Test", model=None, config=config_one)
        
        assert result_zero is not None
        assert result_one is not None
    
    def test_translate_trace_id_uniqueness(self):
        """Verify trace_id is generated and has expected format"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("Test", model=None, config=None)
        
        assert result.trace_id is not None
        assert len(result.trace_id) == 12
        assert all(c in '0123456789abcdef' for c in result.trace_id)
    
    def test_translate_all_registers(self):
        """Test translation with each register type as detected and target"""
        transmogrifier = Transmogrifier()
        
        for register in Register:
            config = TranslationConfig(target_register=register)
            result = transmogrifier.translate("Test", model="model-x", config=config)
            
            assert result.detected_register in Register.__members__.values()
            assert result.target_register in Register.__members__.values()
            assert result.target_register == register


class TestErrorHandling:
    """Test suite for error cases"""
    
    def test_translate_invalid_register_from_profile_best_register(self):
        """ValueError when profile.best_register is invalid string not in Register enum"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register='invalid_register')
        cache.add_profile("model-x", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        
        with pytest.raises(ValueError):
            transmogrifier.translate("Test", model="model-x", config=None)
    
    def test_translate_invalid_register_from_profile_task_register(self):
        """ValueError when profile.best_register_for_task returns invalid string"""
        cache = MockProfileCache()
        profile = MockModelProfile(
            best_register=Register.direct,
            task_registers={'task': 'not_a_register'}
        )
        cache.add_profile("model-x", profile)
        
        config = TranslationConfig(task_aware=True)
        transmogrifier = Transmogrifier(profile_cache=cache, config=config)
        
        with pytest.raises(ValueError):
            transmogrifier.translate("Test", model="model-x", config=config)
    
    def test_translate_invalid_register_multiple_invalid_strings(self):
        """Test various invalid register strings to ensure ValueError is raised"""
        invalid_strings = ['foo', 'bar', 'DIRECT', 'Direct', '', 'casual_bad', '123']
        
        for invalid_str in invalid_strings:
            cache = MockProfileCache()
            profile = MockModelProfile(best_register=invalid_str)
            cache.add_profile("model-x", profile)
            
            transmogrifier = Transmogrifier(profile_cache=cache)
            
            with pytest.raises(ValueError):
                transmogrifier.translate("Test", model="model-x", config=None)


class TestConfigPrecedence:
    """Test suite for configuration precedence logic"""
    
    def test_config_overrides_profile(self):
        """Config target_register should override profile best_register"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.casual)
        cache.add_profile("model-x", profile)
        
        config = TranslationConfig(target_register=Register.technical)
        transmogrifier = Transmogrifier(profile_cache=cache)
        result = transmogrifier.translate("Test", model="model-x", config=config)
        
        assert result.target_register == Register.technical
    
    def test_profile_used_when_no_config_override(self):
        """Profile best_register should be used when config has no target_register"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.academic)
        cache.add_profile("model-x", profile)
        
        config = TranslationConfig(target_register=None)
        transmogrifier = Transmogrifier(profile_cache=cache)
        result = transmogrifier.translate("Test", model="model-x", config=config)
        
        assert result.target_register == Register.academic
    
    def test_default_used_when_no_config_or_profile(self):
        """Default Register.direct should be used when no config or profile available"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("Test", model=None, config=None)
        
        assert result.target_register == Register.direct
    
    def test_task_register_overrides_best_register(self):
        """When task_aware, profile.best_register_for_task should override best_register"""
        cache = MockProfileCache()
        profile = MockModelProfile(
            best_register=Register.direct,
            task_registers={'creative_writing': Register.narrative}
        )
        cache.add_profile("model-x", profile)
        
        config = TranslationConfig(task_aware=True)
        transmogrifier = Transmogrifier(profile_cache=cache, config=config)
        result = transmogrifier.translate("Write a story", model="model-x", config=config)
        
        assert result.target_register == Register.narrative


class TestPropertyInvariants:
    """Test suite for property invariants"""
    
    def test_invariant_translation_level_ordering(self):
        """Verify TranslationLevel enum ordering: system_prompt < rule_rewrite < llm_translate"""
        assert TranslationLevel.system_prompt.value < TranslationLevel.rule_rewrite.value
        assert TranslationLevel.rule_rewrite.value < TranslationLevel.llm_translate.value
        assert TranslationLevel.system_prompt.value == 1
        assert TranslationLevel.rule_rewrite.value == 2
        assert TranslationLevel.llm_translate.value == 3
    
    def test_invariant_task_spread_threshold(self):
        """Verify task_spread threshold for skipping is exactly 2.0"""
        cache = MockProfileCache()
        
        # Test spread below threshold
        profile_low = MockModelProfile(best_register=Register.direct, is_invariant=True)
        profile_low.task_spread = 1.99
        cache.add_profile("model-low", profile_low)
        
        # Test spread at threshold
        profile_at = MockModelProfile(best_register=Register.direct, is_invariant=True)
        profile_at.task_spread = 2.0
        cache.add_profile("model-at", profile_at)
        
        # Test spread above threshold
        profile_high = MockModelProfile(best_register=Register.direct, is_invariant=True)
        profile_high.task_spread = 2.01
        cache.add_profile("model-high", profile_high)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        
        result_low = transmogrifier.translate("Test", model="model-low", config=None)
        result_at = transmogrifier.translate("Test", model="model-at", config=None)
        result_high = transmogrifier.translate("Test", model="model-high", config=None)
        
        assert result_low.skipped == True
        assert result_at.skipped == False
        assert result_high.skipped == False
    
    def test_invariant_system_prompt_always_generated(self):
        """System prompt is always generated regardless of skip status"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.direct, is_invariant=True)
        profile.task_spread = 1.0
        cache.add_profile("model-x", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        result = transmogrifier.translate("Test", model="model-x", config=None)
        
        assert result.skipped == True
        assert result.system_prompt is not None
        assert isinstance(result.system_prompt, str)
        assert len(result.system_prompt) > 0
    
    def test_invariant_elapsed_ms_always_measured(self):
        """elapsed_ms is always measured and populated in result"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("Test", model=None, config=None)
        
        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0
        assert isinstance(result.elapsed_ms, float)
    
    def test_invariant_trace_id_format(self):
        """trace_id is generated via uuid.uuid4().hex[:12] for each TranslationResult"""
        transmogrifier = Transmogrifier()
        
        results = [transmogrifier.translate("Test", model=None, config=None) for _ in range(10)]
        
        for result in results:
            assert result.trace_id is not None
            assert len(result.trace_id) == 12
            assert all(c in '0123456789abcdef' for c in result.trace_id)
        
        # Verify uniqueness (with high probability)
        trace_ids = [r.trace_id for r in results]
        assert len(set(trace_ids)) == len(trace_ids)
    
    def test_invariant_skipped_implies_skip_reason(self):
        """When skipped is True, skip_reason must be set"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.direct, is_invariant=True)
        profile.task_spread = 1.0
        cache.add_profile("model-x", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        result = transmogrifier.translate("Test", model="model-x", config=None)
        
        if result.skipped:
            assert result.skip_reason is not None
            assert isinstance(result.skip_reason, str)
            assert len(result.skip_reason) > 0
    
    def test_invariant_input_text_preserved(self):
        """result.input_text should always equal the input text parameter"""
        transmogrifier = Transmogrifier()
        
        test_texts = ["Hello", "", "Unicode 🌍", "a" * 1000, "   spaces   "]
        
        for text in test_texts:
            result = transmogrifier.translate(text, model=None, config=None)
            assert result.input_text == text
    
    def test_invariant_detected_and_target_are_valid_registers(self):
        """detected_register and target_register must always be valid Register enum values"""
        transmogrifier = Transmogrifier()
        result = transmogrifier.translate("Test", model=None, config=None)
        
        assert isinstance(result.detected_register, Register)
        assert isinstance(result.target_register, Register)
        assert result.detected_register in Register.__members__.values()
        assert result.target_register in Register.__members__.values()
    
    def test_invariant_same_register_implies_system_prompt_level(self):
        """When detected == target, level_applied must be system_prompt"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.direct)
        cache.add_profile("model-direct", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        # Use text that will be detected as direct
        result = transmogrifier.translate("direct text", model="model-direct", config=None)
        
        if result.detected_register == result.target_register:
            assert result.level_applied == TranslationLevel.system_prompt
    
    def test_invariant_different_register_implies_rule_rewrite_level(self):
        """When detected != target and not skipped, level_applied must be rule_rewrite"""
        cache = MockProfileCache()
        profile = MockModelProfile(best_register=Register.technical)
        cache.add_profile("model-tech", profile)
        
        transmogrifier = Transmogrifier(profile_cache=cache)
        # Use text that will be detected as casual
        result = transmogrifier.translate("casual text", model="model-tech", config=None)
        
        if not result.skipped and result.detected_register != result.target_register:
            assert result.level_applied == TranslationLevel.rule_rewrite


class TestIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_end_to_end_translation_workflow(self):
        """End-to-end test with realistic profile cache and configuration"""
        # Setup
        cache = MockProfileCache()
        gpt4_profile = MockModelProfile(
            best_register=Register.technical,
            is_invariant=False
        )
        cache.add_profile("gpt-4", gpt4_profile)
        
        config = TranslationConfig(
            semantic_threshold=0.85,
            task_aware=True
        )
        
        transmogrifier = Transmogrifier(profile_cache=cache, config=config)
        
        # Execute
        result = transmogrifier.translate(
            "Hey, can you help me with this?",
            model="gpt-4",
            config=config
        )
        
        # Verify
        assert result.input_text == "Hey, can you help me with this?"
        assert result.detected_register == Register.casual
        assert result.target_register == Register.technical
        assert result.level_applied == TranslationLevel.rule_rewrite
        assert result.output_text != result.input_text
        assert result.system_prompt is not None
        assert result.elapsed_ms >= 0
        assert result.trace_id is not None
        assert len(result.trace_id) == 12
        assert result.detected_task is not None
    
    def test_multiple_sequential_translations(self):
        """Test multiple translations to verify state consistency"""
        transmogrifier = Transmogrifier()
        
        texts = ["First test", "Second test", "Third test"]
        results = [transmogrifier.translate(text) for text in texts]
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.input_text == texts[i]
            assert result.trace_id is not None
            assert result.elapsed_ms >= 0
        
        # Verify trace IDs are unique
        trace_ids = [r.trace_id for r in results]
        assert len(set(trace_ids)) == 3
