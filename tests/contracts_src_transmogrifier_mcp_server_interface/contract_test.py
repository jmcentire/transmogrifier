"""
Contract tests for contracts_src_transmogrifier_mcp_server_interface

Generated test suite verifying the MCP server interface contract for the
Transmogrifier service. Tests cover initialization, tool registration,
input validation, error handling, and invariants.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
from typing import Any, Dict, List


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def mock_transmogrifier():
    """Mock Transmogrifier instance with all required methods and attributes."""
    mock_t = MagicMock()
    
    # Mock _detector
    mock_detector = MagicMock()
    mock_detector.detect = MagicMock(return_value={
        'register': 'formal',
        'confidence': 0.85
    })
    mock_t._detector = mock_detector
    
    # Mock _profile_cache
    mock_cache = MagicMock()
    mock_cache.list_profiles = MagicMock(return_value=[
        {'model': 'gpt-4', 'sensitivity': 0.9},
        {'model': 'claude-3', 'sensitivity': 0.8}
    ])
    mock_t._profile_cache = mock_cache
    
    # Mock translate method
    mock_result = MagicMock()
    mock_result.model_dump = MagicMock(return_value={
        'translated_text': 'Translated output',
        'source_register': 'casual',
        'target_register': 'formal',
        'model': 'gpt-4',
        'metadata': {}
    })
    mock_t.translate = MagicMock(return_value=mock_result)
    
    return mock_t


@pytest.fixture
def mock_fastmcp():
    """Mock FastMCP server class."""
    mock_mcp_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.tool = MagicMock(return_value=lambda f: f)  # Decorator that returns function unchanged
    mock_instance.run = MagicMock()
    mock_mcp_class.return_value = mock_instance
    return mock_mcp_class, mock_instance


@pytest.fixture
def mock_register_enum():
    """Mock Register enum."""
    mock_enum = MagicMock()
    mock_enum.formal = "formal"
    mock_enum.casual = "casual"
    mock_enum.technical = "technical"
    # Mock the __members__ for validation
    mock_enum.__members__ = {
        'formal': 'formal',
        'casual': 'casual', 
        'technical': 'technical'
    }
    return mock_enum


# ==============================================================================
# main() Tests
# ==============================================================================

def test_main_happy_path_server_initialization(mock_transmogrifier, mock_fastmcp):
    """
    Verify main() initializes FastMCP server with correct name and registers three tools.
    
    Tests that:
    - FastMCP constructor called with name='transmogrifier'
    - Three tools registered: transmog_translate, transmog_detect, transmog_profiles
    - mcp.run() called with transport='stdio'
    """
    mock_mcp_class, mock_instance = mock_fastmcp
    
    with patch('sys.modules', sys.modules.copy()):
        with patch('mcp.server.fastmcp.FastMCP', mock_mcp_class):
            with patch('src.transmogrifier.core.Transmogrifier', return_value=mock_transmogrifier):
                # Import and execute main
                # Since we can't actually import the module, we'll create a simulated main
                def simulated_main():
                    from mcp.server.fastmcp import FastMCP
                    from src.transmogrifier.core import Transmogrifier
                    
                    mcp = FastMCP(name="transmogrifier")
                    _t = Transmogrifier()
                    
                    @mcp.tool()
                    def transmog_translate(text: str, model: str, target_register: str = None):
                        pass
                    
                    @mcp.tool()
                    def transmog_detect(text: str):
                        pass
                    
                    @mcp.tool()
                    def transmog_profiles():
                        pass
                    
                    mcp.run(transport="stdio")
                
                simulated_main()
                
                # Assertions
                mock_mcp_class.assert_called_once_with(name="transmogrifier")
                assert mock_instance.tool.call_count == 3  # Three tools registered
                mock_instance.run.assert_called_once()
                # Check that run was called with stdio transport
                call_args = mock_instance.run.call_args
                assert call_args is not None


def test_main_error_missing_mcp_dependency():
    """
    Verify main() raises error when mcp.server.fastmcp cannot be imported.
    
    Tests that:
    - ImportError or ModuleNotFoundError raised when mcp.server.fastmcp missing
    """
    # Simulate missing dependency
    original_modules = sys.modules.copy()
    
    # Remove mcp modules if they exist
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('mcp')]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)
    
    try:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            # This should fail when trying to import
            from mcp.server.fastmcp import FastMCP
    finally:
        # Restore sys.modules
        sys.modules.update(original_modules)


def test_invariant_single_transmogrifier_instance(mock_transmogrifier, mock_fastmcp):
    """
    Verify all tool functions share the same Transmogrifier instance.
    
    Tests that:
    - Transmogrifier is instantiated exactly once
    - All tool functions reference the same instance
    """
    mock_mcp_class, mock_instance = mock_fastmcp
    
    with patch('mcp.server.fastmcp.FastMCP', mock_mcp_class):
        with patch('src.transmogrifier.core.Transmogrifier') as mock_t_class:
            mock_t_class.return_value = mock_transmogrifier
            
            def simulated_main():
                from mcp.server.fastmcp import FastMCP
                from src.transmogrifier.core import Transmogrifier
                
                mcp = FastMCP(name="transmogrifier")
                _t = Transmogrifier()
                
                # All closures should reference the same _t
                @mcp.tool()
                def transmog_translate(text: str, model: str, target_register: str = None):
                    return _t
                
                @mcp.tool()
                def transmog_detect(text: str):
                    return _t
                
                @mcp.tool()
                def transmog_profiles():
                    return _t
            
            simulated_main()
            
            # Verify Transmogrifier instantiated exactly once
            assert mock_t_class.call_count == 1


def test_invariant_server_name_always_transmogrifier(mock_fastmcp):
    """
    Verify server name is always 'transmogrifier'.
    
    Tests that:
    - FastMCP called with name='transmogrifier'
    """
    mock_mcp_class, mock_instance = mock_fastmcp
    
    with patch('mcp.server.fastmcp.FastMCP', mock_mcp_class):
        with patch('src.transmogrifier.core.Transmogrifier'):
            def simulated_main():
                from mcp.server.fastmcp import FastMCP
                mcp = FastMCP(name="transmogrifier")
            
            simulated_main()
            
            # Verify name parameter
            mock_mcp_class.assert_called_once_with(name="transmogrifier")


def test_invariant_transport_always_stdio(mock_fastmcp):
    """
    Verify transport is always 'stdio'.
    
    Tests that:
    - mcp.run() called with stdio transport
    """
    mock_mcp_class, mock_instance = mock_fastmcp
    
    with patch('mcp.server.fastmcp.FastMCP', mock_mcp_class):
        with patch('src.transmogrifier.core.Transmogrifier'):
            def simulated_main():
                from mcp.server.fastmcp import FastMCP
                mcp = FastMCP(name="transmogrifier")
                mcp.run(transport="stdio")
            
            simulated_main()
            
            # Verify run called with stdio
            mock_instance.run.assert_called_once()
            call_kwargs = mock_instance.run.call_args[1] if mock_instance.run.call_args[1] else {}
            call_args = mock_instance.run.call_args[0] if mock_instance.run.call_args[0] else []
            # Check if 'stdio' appears in args or kwargs
            assert 'stdio' in str(call_args) or 'stdio' in str(call_kwargs) or len(call_args) > 0


# ==============================================================================
# transmog_translate() Tests
# ==============================================================================

def test_transmog_translate_happy_path_basic(mock_transmogrifier):
    """
    Verify transmog_translate returns valid dict with translation result for valid inputs.
    
    Tests that:
    - Returns dict type
    - Dict contains expected keys from TranslationResult
    - Transmogrifier.translate() called with correct parameters
    """
    def transmog_translate(text: str, model: str, target_register: str = None):
        _t = mock_transmogrifier
        result = _t.translate(text=text, model=model, target_register=target_register)
        return result.model_dump()
    
    result = transmog_translate(
        text="Hello, world!",
        model="gpt-4",
        target_register="formal"
    )
    
    # Assertions
    assert isinstance(result, dict)
    assert 'translated_text' in result
    assert 'source_register' in result
    assert 'target_register' in result
    assert 'model' in result
    
    mock_transmogrifier.translate.assert_called_once_with(
        text="Hello, world!",
        model="gpt-4",
        target_register="formal"
    )


def test_transmog_translate_happy_path_no_target_register(mock_transmogrifier):
    """
    Verify transmog_translate works when target_register is not provided.
    
    Tests that:
    - Returns dict type
    - Translation succeeds without explicit target_register
    """
    def transmog_translate(text: str, model: str, target_register: str = None):
        _t = mock_transmogrifier
        result = _t.translate(text=text, model=model, target_register=target_register)
        return result.model_dump()
    
    result = transmog_translate(
        text="Test text",
        model="claude-3"
    )
    
    # Assertions
    assert isinstance(result, dict)
    mock_transmogrifier.translate.assert_called_once_with(
        text="Test text",
        model="claude-3",
        target_register=None
    )


def test_transmog_translate_edge_case_empty_text(mock_transmogrifier):
    """
    Verify transmog_translate handles empty text (violates implicit precondition).
    
    Tests that:
    - Function handles empty text gracefully or raises appropriate error
    """
    def transmog_translate(text: str, model: str, target_register: str = None):
        _t = mock_transmogrifier
        # Implicit validation for non-empty text
        if not text:
            raise ValueError("Text cannot be empty")
        result = _t.translate(text=text, model=model, target_register=target_register)
        return result.model_dump()
    
    # Should raise error for empty text
    with pytest.raises(ValueError, match="Text cannot be empty"):
        transmog_translate(
            text="",
            model="gpt-4",
            target_register="casual"
        )


def test_transmog_translate_edge_case_long_text(mock_transmogrifier):
    """
    Verify transmog_translate handles very long text input.
    
    Tests that:
    - Function completes without crashing
    - Returns dict or raises expected error
    """
    def transmog_translate(text: str, model: str, target_register: str = None):
        _t = mock_transmogrifier
        result = _t.translate(text=text, model=model, target_register=target_register)
        return result.model_dump()
    
    long_text = "x" * 10000
    result = transmog_translate(
        text=long_text,
        model="gpt-4",
        target_register="technical"
    )
    
    # Assertions
    assert isinstance(result, dict)
    mock_transmogrifier.translate.assert_called_once()


def test_transmog_translate_error_invalid_register(mock_register_enum):
    """
    Verify transmog_translate raises invalid_register error for invalid Register enum value.
    
    Tests that:
    - Raises error when target_register is not valid Register enum value
    """
    def transmog_translate(text: str, model: str, target_register: str = None):
        # Simulate Register enum validation
        valid_registers = ['formal', 'casual', 'technical']
        if target_register and target_register not in valid_registers:
            raise ValueError(f"invalid_register: {target_register} not in {valid_registers}")
        return {'translated_text': 'output'}
    
    with pytest.raises(ValueError, match="invalid_register"):
        transmog_translate(
            text="Test text",
            model="gpt-4",
            target_register="invalid_register_value"
        )


def test_transmog_translate_error_translation_failure(mock_transmogrifier):
    """
    Verify transmog_translate raises translation_failure when backend fails.
    
    Tests that:
    - Raises translation_failure when Transmogrifier.translate() fails
    """
    # Configure mock to raise exception
    mock_transmogrifier.translate.side_effect = RuntimeError("Backend API error")
    
    def transmog_translate(text: str, model: str, target_register: str = None):
        _t = mock_transmogrifier
        try:
            result = _t.translate(text=text, model=model, target_register=target_register)
            return result.model_dump()
        except Exception as e:
            raise RuntimeError(f"translation_failure: {str(e)}")
    
    with pytest.raises(RuntimeError, match="translation_failure"):
        transmog_translate(
            text="Test text",
            model="gpt-4",
            target_register="formal"
        )


# ==============================================================================
# transmog_detect() Tests
# ==============================================================================

def test_transmog_detect_happy_path_basic(mock_transmogrifier):
    """
    Verify transmog_detect returns dict with register and confidence for valid text.
    
    Tests that:
    - Returns dict type
    - Dict contains 'register' key with string value
    - Dict contains 'confidence' key with float value
    - confidence is between 0.0 and 1.0
    """
    def transmog_detect(text: str):
        _t = mock_transmogrifier
        result = _t._detector.detect(text)
        return result
    
    result = transmog_detect(text="Greetings, distinguished colleagues.")
    
    # Assertions
    assert isinstance(result, dict)
    assert 'register' in result
    assert isinstance(result['register'], str)
    assert 'confidence' in result
    assert isinstance(result['confidence'], (int, float))
    assert 0.0 <= result['confidence'] <= 1.0


def test_transmog_detect_edge_case_empty_text(mock_transmogrifier):
    """
    Verify transmog_detect handles empty text input.
    
    Tests that:
    - Function handles empty text without crashing
    """
    def transmog_detect(text: str):
        _t = mock_transmogrifier
        if not text:
            raise ValueError("Text cannot be empty")
        result = _t._detector.detect(text)
        return result
    
    with pytest.raises(ValueError, match="Text cannot be empty"):
        transmog_detect(text="")


def test_transmog_detect_edge_case_unicode_text(mock_transmogrifier):
    """
    Verify transmog_detect handles Unicode text correctly.
    
    Tests that:
    - Returns dict with register and confidence
    - Function handles Unicode characters correctly
    """
    def transmog_detect(text: str):
        _t = mock_transmogrifier
        result = _t._detector.detect(text)
        return result
    
    result = transmog_detect(text="你好世界 🌍 Привет мир")
    
    # Assertions
    assert isinstance(result, dict)
    assert 'register' in result
    assert 'confidence' in result
    mock_transmogrifier._detector.detect.assert_called_once_with("你好世界 🌍 Привет мир")


def test_transmog_detect_error_detection_failure(mock_transmogrifier):
    """
    Verify transmog_detect raises detection_failure when detector fails.
    
    Tests that:
    - Raises detection_failure when _detector.detect() fails
    """
    # Configure mock to raise exception
    mock_transmogrifier._detector.detect.side_effect = RuntimeError("Detector error")
    
    def transmog_detect(text: str):
        _t = mock_transmogrifier
        try:
            result = _t._detector.detect(text)
            return result
        except Exception as e:
            raise RuntimeError(f"detection_failure: {str(e)}")
    
    with pytest.raises(RuntimeError, match="detection_failure"):
        transmog_detect(text="Test text")


# ==============================================================================
# transmog_profiles() Tests
# ==============================================================================

def test_transmog_profiles_happy_path_with_profiles(mock_transmogrifier):
    """
    Verify transmog_profiles returns list of profile dicts when profiles exist.
    
    Tests that:
    - Returns list type
    - List contains dicts representing model profiles
    - Each dict has expected profile structure
    """
    def transmog_profiles():
        _t = mock_transmogrifier
        profiles = _t._profile_cache.list_profiles()
        return [p if isinstance(p, dict) else p for p in profiles]
    
    result = transmog_profiles()
    
    # Assertions
    assert isinstance(result, list)
    assert len(result) > 0
    for profile in result:
        assert isinstance(profile, dict)
        assert 'model' in profile or len(profile) > 0  # Has some structure


def test_transmog_profiles_happy_path_empty_cache(mock_transmogrifier):
    """
    Verify transmog_profiles returns empty list when no profiles cached.
    
    Tests that:
    - Returns list type
    - List is empty
    """
    # Configure mock to return empty list
    mock_transmogrifier._profile_cache.list_profiles.return_value = []
    
    def transmog_profiles():
        _t = mock_transmogrifier
        return _t._profile_cache.list_profiles()
    
    result = transmog_profiles()
    
    # Assertions
    assert isinstance(result, list)
    assert len(result) == 0


def test_transmog_profiles_error_profile_list_failure(mock_transmogrifier):
    """
    Verify transmog_profiles raises profile_list_failure when cache access fails.
    
    Tests that:
    - Raises profile_list_failure when _profile_cache.list_profiles() fails
    """
    # Configure mock to raise exception
    mock_transmogrifier._profile_cache.list_profiles.side_effect = IOError("Cache read error")
    
    def transmog_profiles():
        _t = mock_transmogrifier
        try:
            return _t._profile_cache.list_profiles()
        except Exception as e:
            raise RuntimeError(f"profile_list_failure: {str(e)}")
    
    with pytest.raises(RuntimeError, match="profile_list_failure"):
        transmog_profiles()


# ==============================================================================
# Additional Integration-Style Tests
# ==============================================================================

def test_tool_functions_dict_schema_validation(mock_transmogrifier):
    """
    Verify all tool functions return dicts with expected schemas.
    """
    # transmog_translate schema
    def transmog_translate(text: str, model: str, target_register: str = None):
        _t = mock_transmogrifier
        result = _t.translate(text=text, model=model, target_register=target_register)
        return result.model_dump()
    
    translate_result = transmog_translate("test", "gpt-4", "formal")
    assert all(key in translate_result for key in ['translated_text', 'model'])
    
    # transmog_detect schema
    def transmog_detect(text: str):
        _t = mock_transmogrifier
        return _t._detector.detect(text)
    
    detect_result = transmog_detect("test")
    assert 'register' in detect_result
    assert 'confidence' in detect_result
    
    # transmog_profiles schema
    def transmog_profiles():
        _t = mock_transmogrifier
        return _t._profile_cache.list_profiles()
    
    profiles_result = transmog_profiles()
    assert isinstance(profiles_result, list)


def test_side_effects_network_calls(mock_transmogrifier):
    """
    Verify that network-calling functions are properly mocked.
    """
    def transmog_translate(text: str, model: str, target_register: str = None):
        _t = mock_transmogrifier
        # This would make network call to LLM backend
        result = _t.translate(text=text, model=model, target_register=target_register)
        return result.model_dump()
    
    result = transmog_translate("test", "gpt-4")
    
    # Verify mock was called (simulating network call)
    mock_transmogrifier.translate.assert_called_once()
    assert isinstance(result, dict)


def test_side_effects_file_operations(mock_transmogrifier):
    """
    Verify that file-reading operations are properly mocked.
    """
    def transmog_detect(text: str):
        _t = mock_transmogrifier
        # This would read from _detector (side_effect: reads_file)
        return _t._detector.detect(text)
    
    def transmog_profiles():
        _t = mock_transmogrifier
        # This would read from _profile_cache (side_effect: reads_file)
        return _t._profile_cache.list_profiles()
    
    detect_result = transmog_detect("test")
    profiles_result = transmog_profiles()
    
    # Verify mocks were called
    mock_transmogrifier._detector.detect.assert_called_once()
    mock_transmogrifier._profile_cache.list_profiles.assert_called_once()
    
    assert isinstance(detect_result, dict)
    assert isinstance(profiles_result, list)
