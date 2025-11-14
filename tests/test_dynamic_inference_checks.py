import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.human_llm import HumanLLM
from utils.llm_utils import InferenceTracking, InferenceCheck


def make_stub_llm(tmp_config, **kwargs):
    """Create a HumanLLM instance with minimal dependencies for testing"""
    default_kwargs = {
        'llmORchains_list': {'default_llm': MagicMock(), 'premium_llm': MagicMock()},
        'dynamic_llm_config': tmp_config
    }
    default_kwargs.update(kwargs)
    
    with patch('utils.human_llm.HumanLLMConfig'), \
         patch.object(HumanLLM, 'set_print_color'), \
         patch.object(HumanLLM, 'configure_output_schema'), \
         patch.object(HumanLLM, 'set_default_llmORchain'), \
         patch.object(HumanLLM, 'set_premium_llmORchain'):
        llm = HumanLLM(**default_kwargs)
    return llm


def check_a(text, *, output_id=None):
    """Test check function A"""
    return {"a": True, "output_id": output_id}


def check_b(text, *, output_id=None):
    """Test check function B"""
    return {"b": True, "output_id": output_id}


def fake_parse(code, *, output_id=None):
    """Simulate parsed_ok True and parsed dict used by run_tests"""
    return (True, {"ast": "ok", "main": "fn"})


def fake_run_tests(code, *, output_id=None):
    """Simulate test execution results"""
    return {"passed": True, "output_id": output_id}


class TestDynamicInferenceChecks:
    
    def test_register_checks_and_order(self):
        """Test that checks are registered with proper ordering"""
        cfg = {
            "inference_checks": [
                {"name": "A", "callable": "tests.test_dynamic_inference_checks:check_a", "order": 20},
                {"name": "B", "callable": "tests.test_dynamic_inference_checks:check_b", "order": 10}
            ]
        }
        llm = make_stub_llm(cfg)
        
        # Assert ordering B, A in inference_tracking.inference_checks
        keys = list(llm.inference_tracking.inference_checks.keys())
        assert "B" in keys
        assert "A" in keys
        # B should come before A due to lower order value
        assert keys.index("B") < keys.index("A")
    
    def test_register_checks_with_method(self):
        """Test registering checks using method attribute"""
        # Add test method to mock object
        def test_method(self, text, *, output_id=None):
            return {"method_called": True}
        
        cfg = {
            "inference_checks": [
                {"name": "TestMethod", "method": "test_method", "order": 10}
            ]
        }
        
        llm = make_stub_llm(cfg)
        # Add the method after initialization
        llm.test_method = test_method.__get__(llm, HumanLLM)
        # Re-register to pick up the method
        llm._register_dynamic_inference_checks()
        
        assert "TestMethod" in llm.inference_tracking.inference_checks
    
    def test_register_checks_with_enabled_false(self):
        """Test that checks can be registered but disabled"""
        cfg = {
            "inference_checks": [
                {"name": "DisabledCheck", "callable": "tests.test_dynamic_inference_checks:check_a", "enabled": False}
            ]
        }
        llm = make_stub_llm(cfg)
        
        # Check should be registered but excluded
        assert "DisabledCheck" in llm.inference_tracking.inference_checks
        assert "DisabledCheck" in llm.inference_tracking.excluded_inference_checks
    
    def test_register_checks_with_kwargs(self):
        """Test that checks can be registered with kwargs using partial"""
        def check_with_args(text, arg1="default", arg2=42, *, output_id=None):
            return {"arg1": arg1, "arg2": arg2, "output_id": output_id}
        
        # Manually register the function to avoid import issues in tests
        import tests.test_dynamic_inference_checks
        tests.test_dynamic_inference_checks.check_with_args = check_with_args
        
        cfg = {
            "inference_checks": [
                {
                    "name": "CheckWithArgs", 
                    "callable": "tests.test_dynamic_inference_checks:check_with_args",
                    "kwargs": {"arg1": "custom", "arg2": 100}
                }
            ]
        }
        llm = make_stub_llm(cfg)
        
        assert "CheckWithArgs" in llm.inference_tracking.inference_checks
        # Run the check to verify kwargs are applied
        result = llm.run_manage_inference_checks(0, "test text")
        assert result["CheckWithArgs"]["arg1"] == "custom"
        assert result["CheckWithArgs"]["arg2"] == 100
    
    def test_run_post_checks_and_store(self):
        """Test running inference checks through post-inference modifications"""
        cfg = {
            "inference_checks": [
                {"name": "Code Parsing", "callable": "tests.test_dynamic_inference_checks:fake_parse", "order": 10},
                {"name": "Run Tests", "callable": "tests.test_dynamic_inference_checks:fake_run_tests", "order": 20}
            ]
        }
        llm = make_stub_llm(cfg)
        
        # Simulate context & outputs
        class Output:
            def __init__(self, content):
                self.content = content
        
        context = {"llm_outputs": [Output("code_snippet")]}
        mods = {"run_inference_checks": True}
        
        llm._apply_feedback_modifications(mods, context)
        
        # Check that results were stored - it's stored as a dict with output_id as key
        assert hasattr(llm.inference_tracking, 'last_inference_check_results')
        assert 0 in llm.inference_tracking.last_inference_check_results
        results = llm.inference_tracking.last_inference_check_results[0]
        assert "Code Parsing" in results
        assert "Run Tests" in results
        
        # Check that context was updated
        assert "inference_checks" in context
        assert 0 in context["inference_checks"]
    
    def test_run_selected_post_checks(self):
        """Test running only selected inference checks"""
        cfg = {
            "inference_checks": [
                {"name": "Check1", "callable": "tests.test_dynamic_inference_checks:check_a"},
                {"name": "Check2", "callable": "tests.test_dynamic_inference_checks:check_b"}
            ]
        }
        llm = make_stub_llm(cfg)
        
        class Output:
            def __init__(self, content):
                self.content = content
        
        context = {"llm_outputs": [Output("test")]}
        mods = {"post_checks": ["Check1"]}  # Only run Check1
        
        llm._apply_feedback_modifications(mods, context)
        
        results = llm.inference_tracking.last_inference_check_results[0]
        assert "Check1" in results
        assert "Check2" not in results  # Should not be present since not selected
    
    def test_enable_disable_checks_dynamic(self):
        """Test enabling/disabling checks through modifications"""
        cfg = {
            "inference_checks": [
                {"name": "TestCheck", "callable": "tests.test_dynamic_inference_checks:check_a", "enabled": False}
            ]
        }
        llm = make_stub_llm(cfg)
        
        # Initially should be disabled
        assert "TestCheck" in llm.inference_tracking.excluded_inference_checks
        
        # Enable through modifications
        context = {}
        mods = {"enable_checks": ["TestCheck"]}
        llm._apply_modifications(mods, context, "pre_inference")  # Changed to use _apply_modifications
        
        # Should no longer be excluded
        assert "TestCheck" not in llm.inference_tracking.excluded_inference_checks
        
        # Disable again
        mods = {"disable_checks": ["TestCheck"]}
        llm._apply_modifications(mods, context, "pre_inference")  # Changed to use _apply_modifications
        
        # Should be excluded again
        assert "TestCheck" in llm.inference_tracking.excluded_inference_checks
    
    def test_no_checks_config(self):
        """Test that HumanLLM works normally when no inference_checks config is provided"""
        cfg = {}  # No inference_checks
        llm = make_stub_llm(cfg)
        
        # Should work without issues
        context = {"llm_outputs": []}
        mods = {"run_inference_checks": True}
        
        # Should not raise any exceptions
        llm._apply_feedback_modifications(mods, context)
    
    def test_invalid_check_config(self):
        """Test handling of invalid check configurations"""
        cfg = {
            "inference_checks": [
                {"name": "NoFunction"},  # No callable or method
                {"name": "InvalidCallable", "callable": "nonexistent.module:func"},
                {"callable": "tests.test_dynamic_inference_checks:check_a"}  # No name
            ]
        }
        
        # Should not raise exceptions, just log warnings
        llm = make_stub_llm(cfg)
        
        # Only valid checks should be registered
        assert "NoFunction" not in llm.inference_tracking.inference_checks
        assert "InvalidCallable" not in llm.inference_tracking.inference_checks
    
    def test_fallback_attribute_callable(self):
        """Test fallback to attribute lookup for callable"""
        cfg = {
            "inference_checks": [
                {"name": "AttributeCheck", "callable": "test_method"}  # No colon, should fallback to attribute
            ]
        }
        
        llm = make_stub_llm(cfg)
        
        # Add method and re-register
        def test_method(text, *, output_id=None):
            return {"attribute_called": True}
        
        llm.test_method = test_method
        llm._register_dynamic_inference_checks()
        
        assert "AttributeCheck" in llm.inference_tracking.inference_checks
