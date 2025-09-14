import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.human_llm import HumanLLM


class MockOutput:
    def __init__(self, content):
        self.content = content


class TestIntegrationCodingAgent:
    
    @patch('utils.human_llm.HumanLLMConfig')
    @patch.object(HumanLLM, 'set_print_color')
    @patch.object(HumanLLM, 'configure_output_schema')
    @patch.object(HumanLLM, 'set_default_llmORchain')
    @patch.object(HumanLLM, 'set_premium_llmORchain')
    def test_coding_agent_parse_and_test_chain(self, mock_set_premium, mock_set_default, mock_config_schema, mock_color, mock_config):
        """Test full chain with CodingAgent-like workflow using parse and test checks"""
        
        # Configure dynamic checks
        cfg = {
            "inference_checks": [
                {"name": "Code Parsing", "method": "parse_ai_generated_code", "order": 10},
                {"name": "Run Tests", "method": "run_tests_on_code", "order": 20},
                {"name": "Recommend Critics", "method": "generate_instructions_feedback_fn", "order": 30}
            ],
            "run_all_coding_checks": {
                "phase": "post_inference",
                "modifications": {"run_inference_checks": True}
            }
        }
        
        # Create HumanLLM instance
        llm = HumanLLM(
            llmORchains_list={'default_llm': MagicMock(), 'premium_llm': MagicMock()},
            dynamic_llm_config=cfg
        )
        
        # Mock the check methods to return deterministic results
        def mock_parse_code(text, *, output_id=None):
            return (True, {"info": "parsed", "ast": "valid", "main": "function_found"})
        
        def mock_run_tests(text, *, output_id=None):
            return {"passed": True, "test_count": 5, "failures": 0}
        
        def mock_generate_feedback(text, *, output_id=None):
            return {"suggestions": ["consider adding docstrings", "improve error handling"]}
        
        # Attach mock methods to the llm instance
        llm.parse_ai_generated_code = mock_parse_code
        llm.run_tests_on_code = mock_run_tests
        llm.generate_instructions_feedback_fn = mock_generate_feedback
        
        # Manually register the checks since methods were added after initial registration
        llm.add_manage_inference_check("Code Parsing", mock_parse_code)
        llm.add_manage_inference_check("Run Tests", mock_run_tests)  
        llm.add_manage_inference_check("Recommend Critics", mock_generate_feedback)
        
        # Enable the "Recommend Critics" check since it's excluded by default
        llm.include_manage_inference_check(["Recommend Critics"])
        
        # Simulate post-inference scenario
        outputs = [MockOutput("def add(a, b): return a + b")]
        context = {"llm_outputs": outputs}
        mods = {"run_inference_checks": True}
        
        # Run the post-inference modifications
        llm._apply_feedback_modifications(mods, context)
        
        # Verify all checks were run and stored
        assert len(llm.inference_tracking.last_inference_check_results) >= 1
        stored_results = llm.inference_tracking.last_inference_check_results[0]
        
        # Check that all three checks were executed
        assert "Code Parsing" in stored_results
        assert "Run Tests" in stored_results  
        assert "Recommend Critics" in stored_results
        
        # Verify the results contain expected data
        parse_result = stored_results["Code Parsing"]
        assert parse_result[0] is True  # parsed_ok
        assert parse_result[1]["info"] == "parsed"
        
        test_result = stored_results["Run Tests"]
        assert test_result["passed"] is True
        assert test_result["test_count"] == 5
        
        feedback_result = stored_results["Recommend Critics"]
        assert len(feedback_result["suggestions"]) == 2
        
        # Verify context was updated
        assert "inference_checks" in context
        assert 0 in context["inference_checks"]
        assert context["inference_checks"][0] == stored_results

    @patch('utils.human_llm.HumanLLMConfig')
    @patch.object(HumanLLM, 'set_print_color')
    @patch.object(HumanLLM, 'configure_output_schema')
    @patch.object(HumanLLM, 'set_default_llmORchain')
    @patch.object(HumanLLM, 'set_premium_llmORchain')
    def test_selective_check_execution(self, mock_set_premium, mock_set_default, mock_config_schema, mock_color, mock_config):
        """Test running only selected checks in the chain"""
        
        cfg = {
            "inference_checks": [
                {"name": "Code Parsing", "method": "parse_ai_generated_code", "order": 10},
                {"name": "Run Tests", "method": "run_tests_on_code", "order": 20},
                {"name": "Style Check", "method": "check_code_style", "order": 30}
            ]
        }
        
        llm = HumanLLM(
            llmORchains_list={'default_llm': MagicMock(), 'premium_llm': MagicMock()},
            dynamic_llm_config=cfg
        )
        
        # Mock methods
        llm.parse_ai_generated_code = lambda text, *, output_id=None: (True, {"parsed": True})
        llm.run_tests_on_code = lambda text, *, output_id=None: {"passed": True}
        llm.check_code_style = lambda text, *, output_id=None: {"style_score": 8.5}
        
        llm._register_dynamic_inference_checks()
        
        # Run only parsing and testing, skip style check
        outputs = [MockOutput("code")]
        context = {"llm_outputs": outputs}
        mods = {"post_checks": ["Code Parsing", "Run Tests"]}  # Only these two
        
        llm._apply_feedback_modifications(mods, context)
        
        results = llm.inference_tracking.last_inference_check_results[0]
        
        # Should have run only the selected checks
        assert "Code Parsing" in results
        assert "Run Tests" in results
        assert "Style Check" not in results

    @patch('utils.human_llm.HumanLLMConfig')
    @patch.object(HumanLLM, 'set_print_color')
    @patch.object(HumanLLM, 'configure_output_schema')
    @patch.object(HumanLLM, 'set_default_llmORchain')
    @patch.object(HumanLLM, 'set_premium_llmORchain')
    def test_check_chaining_dependency(self, mock_set_premium, mock_set_default, mock_config_schema, mock_color, mock_config):
        """Test that checks run in correct order and results are stored"""
        
        cfg = {
            "inference_checks": [
                {"name": "Parse", "method": "parse_step", "order": 10},
                {"name": "Test", "method": "test_step", "order": 20}
            ]
        }
        
        llm = HumanLLM(
            llmORchains_list={'default_llm': MagicMock(), 'premium_llm': MagicMock()},
            dynamic_llm_config=cfg
        )
        
        # Mock parse step that stores result for next step
        def parse_step(text, *, output_id=None):
            # Store parsing result that test step can use
            return {"parsed_ast": "valid", "functions": ["add", "subtract"]}
        
        def test_step(text, *, output_id=None):
            # Simulate test execution (chaining would require more complex implementation)
            return {"all_passed": True, "test_count": 2}
        
        llm.parse_step = parse_step
        llm.test_step = test_step
        
        # Manually register the checks since methods were added after initial registration
        llm.add_manage_inference_check("Parse", parse_step)
        llm.add_manage_inference_check("Test", test_step)
        
        # Simulate running all checks in one call (more realistic)
        outputs = [MockOutput("def add(a,b): return a+b")]
        context = {"llm_outputs": outputs}
        
        # Run all checks at once
        mods = {"run_inference_checks": True}
        llm._apply_feedback_modifications(mods, context)
        
        results = llm.inference_tracking.last_inference_check_results[0]
        
        # Verify both checks ran in correct order
        assert "Parse" in results
        assert "Test" in results
        
        # Check that checks returned expected data
        assert results["Parse"]["parsed_ast"] == "valid"
        assert results["Test"]["all_passed"] is True
        assert results["Test"]["test_count"] == 2
    
    @patch('utils.human_llm.HumanLLMConfig')
    @patch.object(HumanLLM, 'set_print_color')
    @patch.object(HumanLLM, 'configure_output_schema') 
    @patch.object(HumanLLM, 'set_default_llmORchain')
    @patch.object(HumanLLM, 'set_premium_llmORchain')
    def test_multiple_outputs_processing(self, mock_set_premium, mock_set_default, mock_config_schema, mock_color, mock_config):
        """Test that checks run on multiple outputs correctly"""
        
        cfg = {
            "inference_checks": [
                {"name": "Validate", "method": "validate_output", "order": 10}
            ]
        }
        
        llm = HumanLLM(
            llmORchains_list={'default_llm': MagicMock(), 'premium_llm': MagicMock()},
            dynamic_llm_config=cfg
        )
        
        def validate_output(text, *, output_id=None):
            return {"output_id": output_id, "length": len(text), "valid": len(text) > 5}
        
        llm.validate_output = validate_output
        llm._register_dynamic_inference_checks()
        
        # Create multiple outputs
        outputs = [
            MockOutput("short"),
            MockOutput("this is a longer output"),
            MockOutput("medium length")
        ]
        context = {"llm_outputs": outputs}
        mods = {"run_inference_checks": True}
        
        llm._apply_feedback_modifications(mods, context)
        
        # Should have results for all three outputs
        assert len(llm.inference_tracking.last_inference_check_results) >= 3
        
        for i in range(3):
            results = llm.inference_tracking.last_inference_check_results[i]
            assert "Validate" in results
            assert results["Validate"]["output_id"] == i
            
        # Verify context has all results
        assert len(context["inference_checks"]) == 3
