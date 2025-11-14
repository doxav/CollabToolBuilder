import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from utils.human_llm import DynamicConfigManager, HelpUsageTracker


class TestRenderFeedbackText:
    """Test suite for DynamicConfigManager._render_feedback_text method"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock HumanLLMConfig with common_vectordb"""
        config = Mock()
        config.common_vectordb = Mock()
        config.common_vectordb.populate_few_shot_tags = Mock(return_value="populated few shots")
        return config
    
    @pytest.fixture
    def usage_tracker(self):
        """Create a HelpUsageTracker instance"""
        return HelpUsageTracker()
    
    @pytest.fixture
    def dynamic_manager(self, mock_config, usage_tracker):
        """Create DynamicConfigManager with mocked dependencies"""
        default_config = {'test': 'default'}
        manager = DynamicConfigManager({}, usage_tracker, default_config, human_llm_config=mock_config)
        return manager

    def test_collect_feedback_with_prompt(self, dynamic_manager, mock_config):
        """Test collect_feedback mode with prompt containing few-shots tags"""
        context = {"test": "context"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Here is some text with {{few_shots}} tags"
        }
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Verify the prompt was passed to populate_few_shot_tags
        mock_config.common_vectordb.populate_few_shot_tags.assert_called_once_with(
            "Here is some text with {{few_shots}} tags"
        )
        assert result == "populated few shots"

    def test_collect_feedback_with_prompt_no_method(self, usage_tracker):
        """Test collect_feedback mode when populate_few_shot_tags method doesn't exist"""
        # Create manager with config that doesn't have the method
        class FakeVectorDB:
            pass  # No populate_few_shot_tags method
            
        class FakeConfig:
            def __init__(self):
                self.common_vectordb = FakeVectorDB()
        
        manager = DynamicConfigManager({}, usage_tracker, {})
        manager.config = FakeConfig()
        
        context = {"test": "context"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Here is some text with {{few_shots}} tags"
        }
        
        result = manager._render_feedback_text(context, fb_cfg)
        
        # Should fallback to returning the prompt as-is
        assert result == "Here is some text with {{few_shots}} tags"

    def test_collect_feedback_without_prompt_fallback(self, dynamic_manager):
        """Test collect_feedback mode without prompt falls back to collect_feedback"""
        context = {"test": "context"}
        fb_cfg = {"use": "collect_feedback"}
        
        # Mock collect_feedback to return test data
        test_items = [
            {"content": "test content", "diff": "test diff"},
            {"metrics": "test metrics", "other": "ignored"}
        ]
        with patch.object(dynamic_manager, 'collect_feedback', return_value=test_items):
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Should join content, diff, metrics from all items
        expected = "test content\n\ntest diff\n\ntest metrics"
        assert result == expected

    def test_collect_feedback_empty_fallback(self, dynamic_manager):
        """Test collect_feedback mode with empty items returns 'No feedback'"""
        context = {"test": "context"}
        fb_cfg = {"use": "collect_feedback"}
        
        with patch.object(dynamic_manager, 'collect_feedback', return_value=[]):
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert result == "No feedback"

    def test_custom_fn_basic(self, dynamic_manager):
        """Test custom_fn mode with basic function call"""
        context = {"user": "test_user", "data": {"value": 42}}
        fb_cfg = {
            "use": "custom_fn",
            "fn": "tests.test_render_feedback_text:mock_feedback_function",
            "args": {"message": "hello", "user": "static_user"}
        }
        
        # The mock function will be called via import
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_function = Mock(return_value="custom feedback result")
            mock_module.mock_feedback_function = mock_function
            mock_import.return_value = mock_module
            
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Verify function was called with correct args
        mock_function.assert_called_once_with(message="hello", user="static_user")
        assert result == "custom feedback result"

    def test_custom_fn_with_context_substitution(self, dynamic_manager):
        """Test custom_fn mode with context variable substitution"""
        context = {"user": "test_user", "data": {"value": 42}}
        fb_cfg = {
            "use": "custom_fn", 
            "fn": "tests.test_render_feedback_text:mock_feedback_function",
            "args": {
                "user": "${context.user}",
                "value": "${context.data.value}",
                "static": "unchanged"
            }
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_function = Mock(return_value="substituted result")
            mock_module.mock_feedback_function = mock_function
            mock_import.return_value = mock_module
            
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Verify substitution worked
        mock_function.assert_called_once_with(
            user="test_user", 
            value=42, 
            static="unchanged"
        )
        assert result == "substituted result"

    def test_custom_fn_substitution_missing_keys(self, dynamic_manager):
        """Test custom_fn mode handles missing context keys gracefully"""
        context = {"user": "test_user"}
        fb_cfg = {
            "use": "custom_fn",
            "fn": "tests.test_render_feedback_text:mock_feedback_function", 
            "args": {
                "existing": "${context.user}",
                "missing": "${context.missing.key}"
            }
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_function = Mock(return_value="result with none")
            mock_module.mock_feedback_function = mock_function
            mock_import.return_value = mock_module
            
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Missing key should be None
        mock_function.assert_called_once_with(existing="test_user", missing=None)
        assert result == "result with none"

    def test_custom_fn_non_string_args(self, dynamic_manager):
        """Test custom_fn mode with non-string args (no substitution)"""
        context = {"user": "test_user"}
        fb_cfg = {
            "use": "custom_fn",
            "fn": "tests.test_render_feedback_text:mock_feedback_function",
            "args": {
                "number": 123,
                "boolean": True,
                "list": [1, 2, 3],
                "dict": {"nested": "value"}
            }
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_function = Mock(return_value="non-string result")
            mock_module.mock_feedback_function = mock_function
            mock_import.return_value = mock_module
            
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Non-string args should pass through unchanged
        mock_function.assert_called_once_with(
            number=123,
            boolean=True, 
            list=[1, 2, 3],
            dict={"nested": "value"}
        )
        assert result == "non-string result"

    def test_legacy_fallback_no_use_key(self, dynamic_manager):
        """Test legacy fallback when no 'use' key provided"""
        context = {"test": "context"}
        fb_cfg = {"some": "other", "config": "values"}
        
        test_items = [
            {"content": "legacy content"},
            {"diff": "legacy diff"},
            {"metrics": "legacy metrics"}
        ]
        with patch.object(dynamic_manager, 'collect_feedback', return_value=test_items):
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        expected = "legacy content\n\nlegacy diff\n\nlegacy metrics"
        assert result == expected

    def test_legacy_fallback_none_fb_cfg(self, dynamic_manager):
        """Test legacy fallback when fb_cfg is None"""
        context = {"test": "context"}
        
        test_items = [{"content": "none config content"}]
        with patch.object(dynamic_manager, 'collect_feedback', return_value=test_items):
            result = dynamic_manager._render_feedback_text(context, None)
        
        assert result == "none config content"

    def test_unknown_use_mode_fallback(self, dynamic_manager):
        """Test unknown 'use' mode falls back to legacy behavior"""
        context = {"test": "context"}
        fb_cfg = {"use": "unknown_mode", "other": "config"}
        
        test_items = [{"content": "fallback content"}]
        with patch.object(dynamic_manager, 'collect_feedback', return_value=test_items):
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert result == "fallback content"

    def test_custom_fn_import_error_handling(self, dynamic_manager):
        """Test custom_fn mode handles import errors gracefully"""
        context = {"test": "context"}
        fb_cfg = {
            "use": "custom_fn",
            "fn": "nonexistent.module:function",
            "args": {}
        }
        
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError):
                dynamic_manager._render_feedback_text(context, fb_cfg)

    def test_custom_fn_attribute_error_handling(self, dynamic_manager):
        """Test custom_fn mode handles missing function attributes"""
        context = {"test": "context"} 
        fb_cfg = {
            "use": "custom_fn",
            "fn": "tests.test_render_feedback_text:nonexistent_function",
            "args": {}
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            # Function doesn't exist on module - delete it to ensure AttributeError
            if hasattr(mock_module, 'nonexistent_function'):
                delattr(mock_module, 'nonexistent_function')
            mock_import.return_value = mock_module
            
            with pytest.raises(AttributeError):
                dynamic_manager._render_feedback_text(context, fb_cfg)

    def test_context_substitution_edge_cases(self, dynamic_manager):
        """Test edge cases in context substitution"""
        context = {"nested": {"deep": {"value": "found"}}}
        
        test_cases = [
            # Valid deep nesting
            ("${context.nested.deep.value}", "found"),
            # Invalid format (no closing brace)
            ("${context.nested", "${context.nested"),
            # Empty substitution
            ("${}", "${}"),
            # Non-context prefix
            ("${other.key}", "${other.key}"),
            # Context but no additional path
            ("${context}", context),
        ]
        
        for input_val, expected in test_cases:
            fb_cfg = {
                "use": "custom_fn",
                "fn": "tests.test_render_feedback_text:mock_feedback_function",
                "args": {"test": input_val}
            }
            
            with patch('importlib.import_module') as mock_import:
                mock_module = Mock()
                mock_function = Mock(return_value="test")
                mock_module.mock_feedback_function = mock_function
                mock_import.return_value = mock_module
                
                dynamic_manager._render_feedback_text(context, fb_cfg)
                
                # Check what was actually passed
                call_args = mock_function.call_args[1]
                assert call_args["test"] == expected

    def test_custom_fn_with_actual_function(self, dynamic_manager):
        """Test custom_fn with an actual importable function"""
        context = {"value": "test_value"}
        fb_cfg = {
            "use": "custom_fn",
            "fn": "json:dumps", 
            "args": {"obj": {"context_value": "${context.value}"}}
        }
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # json.dumps should return a JSON string
        import json
        expected = json.dumps({"context_value": "test_value"})
        assert result == expected

    def test_substitution_with_none_context_values(self, dynamic_manager):
        """Test substitution when context contains None values"""
        context = {"user": None, "data": {"value": None}}
        fb_cfg = {
            "use": "custom_fn",
            "fn": "str:upper",  # Using str.upper as a simple function
            "args": {"self": "${context.user}"}
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_function = Mock(return_value="NONE")
            mock_module.upper = mock_function
            mock_import.return_value = mock_module
            
            result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # None should be passed through as None
        mock_function.assert_called_once_with(self=None)
        assert result == "NONE"


# Mock function for testing custom_fn functionality
def mock_feedback_function(**kwargs):
    """Mock function used in custom_fn tests"""
    return f"Mock called with: {json.dumps(kwargs, sort_keys=True)}"


# ------------------------------------------------------------------------------
# Feedback Quality and Availability Tests
# ------------------------------------------------------------------------------

class TestFeedbackAvailabilityScenarios:
    """Test suite to diagnose when feedback is available, poor, or missing"""
    
    @pytest.fixture
    def usage_tracker(self):
        """Create a HelpUsageTracker instance"""
        return HelpUsageTracker()
    
    @pytest.fixture
    def dynamic_manager(self, usage_tracker):
        """Create DynamicConfigManager with empty records"""
        manager = DynamicConfigManager({}, usage_tracker, {})
        manager._records = []  # Ensure clean state
        return manager
    
    def test_no_records_no_context_metrics(self, dynamic_manager):
        """Scenario 1: First call - no records, no context metrics"""
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert result == "No feedback"
    
    def test_no_records_with_context_metrics(self, dynamic_manager):
        """Scenario 2: First call - no records, but timing metrics in context"""
        context = {
            "user_message": "test",
            "inference_time": 1.707936,
            "quality_score": 0.85
        }
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Should fall back to current metrics when no historical data
        assert "Current inference_time: 1.707936s" in result
        assert "Current quality_score: 0.85" in result
    
    def test_records_without_type_field_bug(self, dynamic_manager):
        """Scenario 3: Records exist but missing 'type' field (the bug!)"""
        # Simulate what record_outcome() does (missing type field)
        dynamic_manager._records = [
            {
                "ts": 1234567890,
                # NO "type" field!
                "context": {"user_message": "test"},
                "modifications": {},
                "metrics": {"accuracy": 0.8, "distance": 100}
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # BUG: Records filtered out because type != "auto_eval"
        assert result == "No feedback" or "Current inference_time" in result
    
    def test_records_with_type_field_fixed(self, dynamic_manager):
        """Scenario 4: Records with correct 'type' field (the fix!)"""
        # Proper record with type field
        dynamic_manager._records = [
            {
                "ts": 1234567890,
                "type": "auto_eval",  # ← THE FIX
                "context": {"user_message": "test"},
                "modifications": {},
                "metrics": {"accuracy": 0.8, "distance": 100}
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Should now include metrics!
        assert "accuracy" in result or "0.8" in result or "distance" in result
    
    def test_user_diff_feedback(self, dynamic_manager):
        """Scenario 5: User corrections via log_user_correction"""
        dynamic_manager._records = [
            {
                "type": "user_diff",
                "agent": "test_agent",
                "diff": "Changed output from X to Y because it was wrong",
                "who": "user123",
                "ts": 1234567890
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert "Changed output from X to Y" in result or "diff" in result
    
    def test_human_annotation_feedback(self, dynamic_manager):
        """Scenario 6: Human annotations"""
        dynamic_manager._records = [
            {
                "type": "human_annotation",
                "content": "This answer is incorrect - missing key detail",
                "ts": 1234567890
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert "incorrect" in result or "missing key detail" in result
    
    def test_mixed_feedback_sources(self, dynamic_manager):
        """Scenario 7: Multiple feedback types together"""
        dynamic_manager._records = [
            {
                "type": "user_diff",
                "diff": "User correction: should be 42",
                "ts": 1234567891
            },
            {
                "type": "auto_eval",
                "metrics": {"accuracy": 0.5},
                "ts": 1234567890
            },
            {
                "type": "human_annotation",
                "content": "Needs improvement",
                "ts": 1234567892
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Should include all types
        assert len(result) > 50  # Meaningful feedback
        # Order: user_diffs first, then annotations, then auto_evals
        assert "User correction" in result or "diff" in result
    
    def test_poor_feedback_timing_only(self, dynamic_manager):
        """Scenario 8: Only timing info available (poor feedback)"""
        context = {
            "user_message": "Calculate: 120924 × 9321",
            "inference_time": 1.5
        }
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Poor feedback - optimizer can't learn from timing alone
        assert "inference_time" in result
        assert "accuracy" not in result
        assert "correct" not in result
    
    def test_rich_feedback_with_task_metrics(self, dynamic_manager):
        """Scenario 9: Rich feedback with task-specific metrics"""
        dynamic_manager._records = [
            {
                "type": "auto_eval",
                "context": {
                    "user_message": "Calculate: 120924 × 9321",
                    "expected": 1127253804,
                    "got": "I don't know"
                },
                "metrics": {
                    "accuracy": 0.0,
                    "correctness": False,
                    "distance": 1127253804,
                    "verdict": "INCORRECT - need better prompt"
                },
                "ts": 1234567890
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Rich feedback - optimizer can learn!
        assert "accuracy" in result or "0.0" in result
        assert "INCORRECT" in result or "verdict" in result
    
    def test_feedback_config_include_metrics_flag(self, dynamic_manager):
        """Scenario 10: Explicit include_metrics flag"""
        context = {
            "user_message": "test",
            "inference_time": 2.0,
            "cost": 0.001
        }
        
        # Without flag (default: only if no historical)
        fb_cfg_no_flag = {"use": "collect_feedback"}
        result_no_flag = dynamic_manager._render_feedback_text(context, fb_cfg_no_flag)
        
        # With flag (always include)
        fb_cfg_with_flag = {"use": "collect_feedback", "include_metrics": True}
        result_with_flag = dynamic_manager._render_feedback_text(context, fb_cfg_with_flag)
        
        # Both should include metrics since no historical data
        assert "inference_time" in result_no_flag
        assert "inference_time" in result_with_flag
    
    def test_feedback_config_with_custom_prompt(self, dynamic_manager):
        """Scenario 11: Custom prompt overrides record collection"""
        dynamic_manager._records = [
            {
                "type": "auto_eval",
                "metrics": {"accuracy": 0.9},
                "ts": 1234567890
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Ignore records, use this fixed prompt instead"
        }
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Custom prompt overrides everything
        assert "fixed prompt" in result
        assert "accuracy" not in result  # Records ignored
    
    def test_custom_fn_for_task_specific_evaluation(self, dynamic_manager):
        """Scenario 12: Custom function for immediate evaluation"""
        
        def evaluate_multiplication(user_msg, response, **kwargs):
            import re
            # Parse task
            match = re.search(r"(\d+)\s*×\s*(\d+)", user_msg)
            if not match:
                return "Cannot parse task"
            
            a, b = int(match.group(1)), int(match.group(2))
            expected = a * b
            
            # Parse response
            try:
                result = int(re.search(r'\d+', response).group())
                correct = (result == expected)
                return f"Task: {a}×{b}={expected}\nGot: {result}\nCorrect: {correct}"
            except:
                return f"Task: {a}×{b}={expected}\nGot: {response}\nCorrect: False"
        
        # Register function
        import sys
        sys.modules[__name__].evaluate_multiplication = evaluate_multiplication
        
        context = {
            "user_message": "Calculate: 120924 × 9321",
            "response": "1127253804"
        }
        
        fb_cfg = {
            "use": "custom_fn",
            "fn": f"{__name__}:evaluate_multiplication",
            "args": {
                "user_msg": "${context.user_message}",
                "response": "${context.response}"
            }
        }
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Should evaluate immediately
        assert "120924" in result
        assert "9321" in result
        assert "Correct:" in result


class TestFeedbackQualityAssessment:
    """Tests to categorize feedback quality"""
    
    @pytest.fixture
    def usage_tracker(self):
        return HelpUsageTracker()
    
    @pytest.fixture
    def dynamic_manager(self, usage_tracker):
        manager = DynamicConfigManager({}, usage_tracker, {})
        manager._records = []
        return manager
    
    def test_assess_no_feedback(self, dynamic_manager):
        """Quality: NONE - no feedback available"""
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert result == "No feedback"
        # Quality: NONE - optimizer has nothing to work with
    
    def test_assess_poor_feedback(self, dynamic_manager):
        """Quality: POOR - only timing/cost, no task performance"""
        context = {
            "inference_time": 1.5,
            "cost": 0.001
        }
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Has content but not actionable for task improvement
        assert "inference_time" in result or "cost" in result
        assert "accuracy" not in result
        assert "correct" not in result
        # Quality: POOR - can optimize cost/speed but not task quality
    
    def test_assess_moderate_feedback(self, dynamic_manager):
        """Quality: MODERATE - has metrics but limited context"""
        dynamic_manager._records = [
            {
                "type": "auto_eval",
                "metrics": {"accuracy": 0.3},
                "ts": 1234567890
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Has performance metric but missing context
        assert "accuracy" in result or "0.3" in result
        # Quality: MODERATE - knows performance but not why it's poor
    
    def test_assess_good_feedback(self, dynamic_manager):
        """Quality: GOOD - has metrics with detailed info"""
        dynamic_manager._records = [
            {
                "type": "auto_eval",
                "content": "Task: Calculate 5 × 3\nExpected: 15\nGot: I don't know",
                "metrics": {
                    "accuracy": 0.0,
                    "reason": "Model refused to answer"
                },
                "ts": 1234567890
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Has both performance and context via content field
        assert "accuracy" in result or "0.0" in result
        assert "Expected" in result or "Task:" in result or "Got:" in result
        # Quality: GOOD - optimizer can understand what went wrong
    
    def test_assess_excellent_feedback(self, dynamic_manager):
        """Quality: EXCELLENT - has metrics + context + improvement hints"""
        dynamic_manager._records = [
            {
                "type": "auto_eval",
                "content": "Task: 120924 × 9321 = 1127253804\nGot: 'I don't know'\nProblem: Model doesn't attempt calculation\nSuggestion: Add instruction to show step-by-step work",
                "metrics": {
                    "accuracy": 0.0,
                    "confidence": 0.1
                },
                "ts": 1234567890
            }
        ]
        
        context = {"user_message": "test"}
        fb_cfg = {"use": "collect_feedback"}
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Has everything: performance + context + actionable suggestion
        assert "accuracy" in result or "0.0" in result
        assert "Suggestion" in result or "step-by-step" in result
        # Quality: EXCELLENT - optimizer can take specific action


class TestFeedbackConfigurationStrategies:
    """Test different feedback configuration strategies and their effectiveness"""
    
    @pytest.fixture
    def usage_tracker(self):
        return HelpUsageTracker()
    
    @pytest.fixture
    def dynamic_manager(self, usage_tracker):
        manager = DynamicConfigManager({}, usage_tracker, {})
        manager._records = []
        return manager
    
    def test_strategy_default_collect_feedback(self, dynamic_manager):
        """Strategy 1: Default - collect_feedback with no config"""
        # Pros: Simple, works with record_outcome
        # Cons: Requires type field, delayed feedback
        
        dynamic_manager._records = [
            {
                "type": "auto_eval",
                "metrics": {"score": 0.7},
                "ts": 1234567890
            }
        ]
        
        fb_cfg = {"use": "collect_feedback"}
        result = dynamic_manager._render_feedback_text({}, fb_cfg)
        
        assert "score" in result or "0.7" in result
        # Works if records have type field
    
    def test_strategy_custom_prompt_with_fewshots(self, dynamic_manager):
        """Strategy 2: Custom prompt with few-shot examples"""
        # Pros: Can provide rich context, uses historical data
        # Cons: Requires vector DB, may not have recent metrics
        
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Based on these examples, improve the prompt:\n\nExamples here..."
        }
        
        result = dynamic_manager._render_feedback_text({}, fb_cfg)
        
        assert "examples" in result.lower()
        # Useful for leveraging historical patterns
    
    def test_strategy_custom_fn_immediate_eval(self, dynamic_manager):
        """Strategy 3: Custom function for immediate evaluation"""
        # Pros: Can evaluate during invoke(), most flexible
        # Cons: Requires custom code, more complex
        
        def immediate_evaluator(ctx_data):
            return f"Evaluated: {ctx_data}"
        
        import sys
        sys.modules[__name__].immediate_evaluator = immediate_evaluator
        
        context = {"result": "test_output"}
        fb_cfg = {
            "use": "custom_fn",
            "fn": f"{__name__}:immediate_evaluator",
            "args": {"ctx_data": "${context.result}"}
        }
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert "Evaluated: test_output" in result
        # Best for real-time feedback during POST-phase
    
    def test_strategy_hybrid_historical_plus_current(self, dynamic_manager):
        """Strategy 4: Combine historical records + current metrics"""
        # Pros: Best of both worlds
        # Cons: May be verbose
        
        dynamic_manager._records = [
            {
                "type": "auto_eval",
                "metrics": {"accuracy": 0.6},
                "ts": 1234567890
            }
        ]
        
        context = {
            "inference_time": 2.0,
            "quality_score": 0.8
        }
        
        fb_cfg = {"use": "collect_feedback", "include_metrics": True}
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Should have both historical and current
        assert ("accuracy" in result or "0.6" in result) and "quality_score" in result
        # Most comprehensive feedback
    
    def test_strategy_limits_no_type_field(self, dynamic_manager):
        """Limit 1: Records without type field are ignored"""
        dynamic_manager._records = [
            {
                # Missing type field!
                "metrics": {"accuracy": 0.9},
                "ts": 1234567890
            }
        ]
        
        fb_cfg = {"use": "collect_feedback"}
        result = dynamic_manager._render_feedback_text({}, fb_cfg)
        
        # Bug: Record ignored
        assert result == "No feedback"
    
    def test_strategy_limits_timing_constraint(self, dynamic_manager):
        """Limit 2: POST-phase timing - metrics not yet available"""
        # During invoke(), task metrics aren't computed yet
        # Only inference_time available
        
        context_during_post = {
            "user_message": "Calculate: 5 × 3",
            "inference_time": 1.5
            # accuracy, correctness NOT YET available
        }
        
        fb_cfg = {"use": "collect_feedback"}
        result = dynamic_manager._render_feedback_text(context_during_post, fb_cfg)
        
        # Limited feedback during POST-phase
        assert "inference_time" in result
        assert "accuracy" not in result  # Not computed yet
    
    def test_strategy_workaround_custom_fn(self, dynamic_manager):
        """Workaround 1: Use custom_fn to compute metrics in real-time"""
        
        def compute_metrics_now(user_msg, response):
            # Evaluate immediately during POST-phase
            if "5 × 3" in user_msg and "15" in response:
                return "Correct! Accuracy: 1.0"
            return "Incorrect. Accuracy: 0.0"
        
        import sys
        sys.modules[__name__].compute_metrics_now = compute_metrics_now
        
        context = {
            "user_message": "Calculate: 5 × 3",
            "response": "15"
        }
        
        fb_cfg = {
            "use": "custom_fn",
            "fn": f"{__name__}:compute_metrics_now",
            "args": {
                "user_msg": "${context.user_message}",
                "response": "${context.response}"
            }
        }
        
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        assert "Accuracy: 1.0" in result
        # Overcomes timing constraint!
    
    def test_strategy_workaround_two_pass(self, dynamic_manager):
        """Workaround 2: Two-pass approach - collect then optimize"""
        # Pass 1: Run tasks, record outcomes
        for i in range(3):
            dynamic_manager._records.append({
                "type": "auto_eval",
                "metrics": {"accuracy": i * 0.3},
                "ts": 1234567890 + i
            })
        
        # Pass 2: Aggregate feedback
        context = {}
        fb_cfg = {"use": "collect_feedback"}
        result = dynamic_manager._render_feedback_text(context, fb_cfg)
        
        # Has accumulated feedback
        assert len(result) > 20
        # Can now optimize with full information


# ------------------------------------------------------------------------------
# Integration Tests with Real Few-Shot Functionality
# ------------------------------------------------------------------------------

class TestRenderFeedbackTextIntegration:
    """Integration tests using real few-shot functionality"""
    
    @pytest.fixture
    def real_vectordb_config(self, tmp_path):
        """Create a temporary vector database for testing"""
        from utils.llm_utils import UnifiedVectorDBConfig, CHROMA_DATABASE
        import time
        
        # Simple mock embedding function
        class MockEmbedding:
            def embed_query(self, text):
                return [hash(text) % 100 / 100.0 for _ in range(16)]
            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]
        
        config = UnifiedVectorDBConfig(
            embedding_function=MockEmbedding(),
            collection_name=f"test_feedback_{int(time.time())}",
            persist_directory=str(tmp_path / "test_vectordb"),
            reset_indices=True
        )
        config.db_type = CHROMA_DATABASE
        return config
    
    @pytest.fixture
    def real_vectordb(self, real_vectordb_config):
        """Create actual UnifiedVectorDB instance"""
        from utils.llm_utils import UnifiedVectorDB
        uvdb = UnifiedVectorDB(config=real_vectordb_config, check_db=False)
        yield uvdb
        # Cleanup
        try:
            uvdb.destroy()
        except:
            pass
    
    @pytest.fixture
    def real_config(self, real_vectordb):
        """Create HumanLLMConfig with real vector database"""
        from utils.human_llm_config import HumanLLMConfig
        
        # Reset singleton
        HumanLLMConfig._HumanLLMConfig__instance = None
        
        config = HumanLLMConfig()
        config.common_vectordb = real_vectordb
        config.common_vectordb_config = real_vectordb.config
        config.initialize()
        return config
    
    @pytest.fixture
    def dynamic_manager_with_real_config(self, real_config):
        """Create DynamicConfigManager with real HumanLLMConfig"""
        usage_tracker = HelpUsageTracker()
        default_config = {'test': 'default'}
        manager = DynamicConfigManager({}, usage_tracker, default_config, human_llm_config=real_config)
        return manager

    def test_real_few_shot_population_basic(self, dynamic_manager_with_real_config, real_config):
        """Test real few-shot population with actual vector database"""
        import json
        import time
        
        # Add some test data to the vector database
        test_tasks = [
            {
                "time": "2024-01-01T10:00:00",
                "main_function_name": "add_numbers",
                "task_description": "Add two numbers together",
                "program_code": "def add_numbers(a, b): return a + b"
            },
            {
                "time": "2024-01-01T11:00:00", 
                "main_function_name": "multiply_values",
                "task_description": "Multiply two values",
                "program_code": "def multiply_values(x, y): return x * y"
            }
        ]
        
        # Store tasks in vector database
        for task in test_tasks:
            serialized_task = json.dumps(task)
            tags = {"host": "test_host", "step_id": 1, "task_type": "learnt"}
            real_config.add_learnt_task(serialized_task, tags)
        
        time.sleep(1)  # Allow indexing
        
        # Test few-shot population
        context = {"test": "context"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Here are some examples:\n\nfew_shots:{\"sources\": \"learnt\", \"num\": 2, \"format\": \"json\"}\n\nNow complete the task."
        }
        
        result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
        
        # Verify few-shot tag was replaced with actual content
        assert "few_shots:" not in result
        assert "add_numbers" in result
        assert "multiply_values" in result
        assert "Here are some examples:" in result
        assert "Now complete the task." in result

    def test_real_few_shot_with_multiple_tags(self, dynamic_manager_with_real_config, real_config):
        """Test multiple few-shot tags in a single prompt"""
        import json
        import time
        
        # Add different types of tasks
        learnt_tasks = [
            {
                "time": "2024-01-01T10:00:00",
                "main_function_name": "successful_function",
                "task_description": "A successful implementation",
                "success": True
            }
        ]
        
        failed_tasks = [
            {
                "time": "2024-01-01T09:00:00",
                "main_function_name": "failed_function", 
                "task_description": "A failed implementation",
                "error": "Something went wrong",
                "success": False
            }
        ]
        
        # Store tasks
        for task in learnt_tasks:
            serialized_task = json.dumps(task)
            tags = {"host": "test_host", "step_id": 1, "task_type": "learnt"}
            real_config.add_learnt_task(serialized_task, tags)
        
        for task in failed_tasks:
            serialized_task = json.dumps(task)
            tags = {"host": "test_host", "step_id": 1, "task_type": "failed"}
            real_config.add_failed_task(serialized_task, tags)
        
        time.sleep(1)
        
        # Prompt with multiple few-shot tags
        prompt_with_multiple_tags = '''System: Here are successful examples:
few_shots:{"sources": "learnt", "num": 1, "format": "json"}

And here are failure cases to avoid:
few_shots:{"sources": "failed", "num": 1, "format": "json"}

Complete the task avoiding the failures.'''
        
        context = {"test": "context"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": prompt_with_multiple_tags
        }
        
        result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
        
        # Verify both tags were processed
        assert "few_shots:" not in result
        assert "successful_function" in result
        assert "failed_function" in result
        assert "Here are successful examples:" in result
        assert "failure cases to avoid:" in result

    def test_real_few_shot_empty_database(self, dynamic_manager_with_real_config):
        """Test few-shot processing when database is empty"""
        context = {"test": "context"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Examples: few_shots:{\"sources\": \"learnt\", \"num\": 5}\n\nComplete task."
        }
        
        result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
        
        # Should handle empty database gracefully
        assert "few_shots:" not in result
        assert "Examples:" in result
        assert "Complete task." in result

    def test_real_few_shot_with_context_substitution_integration(self, dynamic_manager_with_real_config, real_config):
        """Test integration of few-shot functionality with context substitution in custom_fn mode"""
        import json
        import time
        
        # Add test data
        test_task = {
            "time": "2024-01-01T10:00:00",
            "main_function_name": "test_function",
            "task_description": "Test function description"
        }
        
        serialized_task = json.dumps(test_task)
        tags = {"host": "test_host", "step_id": 1, "task_type": "learnt"}
        real_config.add_learnt_task(serialized_task, tags)
        
        time.sleep(1)
        
        # Create a custom function that uses few-shot functionality
        def custom_feedback_with_fewshots(user_name, fewshot_prompt):
            # This simulates calling populate_few_shot_tags from within custom function
            processed_prompt = real_config.populate_few_shot_tags(fewshot_prompt)
            return f"User {user_name}: {processed_prompt}"
        
        # Register the function in the global namespace for import
        import sys
        sys.modules[__name__].custom_feedback_with_fewshots = custom_feedback_with_fewshots
        
        context = {"user": "john_doe"}
        fb_cfg = {
            "use": "custom_fn",
            "fn": f"{__name__}:custom_feedback_with_fewshots",
            "args": {
                "user_name": "${context.user}",
                "fewshot_prompt": "Examples: few_shots:{\"sources\": \"learnt\", \"num\": 1, \"format\": \"json\"}"
            }
        }
        
        result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
        
        # Verify both context substitution and few-shot processing worked
        assert "User john_doe:" in result
        assert "test_function" in result
        assert "few_shots:" not in result

    def test_real_few_shot_different_formats(self, dynamic_manager_with_real_config, real_config):
        """Test different few-shot formats with real data"""
        import json
        import time
        
        # Add test data
        test_task = {
            "time": "2024-01-01T10:00:00",
            "main_function_name": "format_test",
            "task_description": "Testing different formats",
            "program_code": "def format_test(): pass"
        }
        
        serialized_task = json.dumps(test_task)
        tags = {"host": "test_host", "step_id": 1, "task_type": "learnt"}
        real_config.add_learnt_task(serialized_task, tags)
        
        time.sleep(1)
        
        # Test different formats
        formats_to_test = ["json", "markdown", "jinja2"]
        
        for format_type in formats_to_test:
            context = {"test": "context"}
            fb_cfg = {
                "use": "collect_feedback",
                "prompt": f"Format test:\n\nfew_shots:{{\"sources\": \"learnt\", \"num\": 1, \"format\": \"{format_type}\"}}\n\nEnd test."
            }
            
            result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
            
            # Basic verification - few-shot tag should be replaced
            assert "few_shots:" not in result, f"Few-shot tag not replaced for format {format_type}"
            assert "format_test" in result, f"Function name not found for format {format_type}"
            assert "Format test:" in result, f"Prompt text not preserved for format {format_type}"
            assert "End test." in result, f"Prompt text not preserved for format {format_type}"

    def test_real_few_shot_malformed_json_in_tag(self, dynamic_manager_with_real_config):
        """Test handling of malformed JSON in few-shot tags with real config"""
        malformed_prompts = [
            'Test: few_shots:{"sources": "learnt", "num": 1 and more',  # Missing closing brace
            'Test: few_shots:{"sources": learnt, "num": 1} more',       # Unquoted value
            'Test: few_shots:{invalid json} more text',                 # Invalid JSON
        ]
        
        for malformed_prompt in malformed_prompts:
            context = {"test": "context"}
            fb_cfg = {
                "use": "collect_feedback",
                "prompt": malformed_prompt
            }
            
            # Should not crash, should handle gracefully
            result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
            
            # Should preserve the malformed tag or handle it gracefully
            assert "Test:" in result
            assert len(result) > 0

    def test_real_few_shot_performance_with_data(self, dynamic_manager_with_real_config, real_config):
        """Test performance with a reasonable amount of data"""
        import json
        import time
        
        # Add multiple tasks
        for i in range(10):
            test_task = {
                "time": f"2024-01-01T{10+i}:00:00",
                "main_function_name": f"function_{i}",
                "task_description": f"Task number {i}",
                "program_code": f"def function_{i}(): return {i}"
            }
            
            serialized_task = json.dumps(test_task)
            tags = {"host": "test_host", "step_id": i, "task_type": "learnt"}
            real_config.add_learnt_task(serialized_task, tags)
        
        time.sleep(2)  # Allow indexing
        
        # Test performance
        start_time = time.time()
        
        context = {"test": "context"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Examples:\n\nfew_shots:{\"sources\": \"learnt\", \"num\": 5, \"format\": \"json\"}\n\nComplete the task."
        }
        
        result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify it completed in reasonable time (adjust threshold as needed)
        assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f}s"
        
        # Verify content was processed correctly
        assert "few_shots:" not in result
        assert "function_" in result  # Should contain at least one function
        assert "Examples:" in result
        assert "Complete the task." in result

    def test_real_few_shot_with_filters(self, dynamic_manager_with_real_config, real_config):
        """Test few-shot functionality with metadata filters"""
        import json
        import time
        
        # Add tasks with different metadata
        tasks_with_metadata = [
            {
                "task": {
                    "time": "2024-01-01T10:00:00",
                    "main_function_name": "easy_task",
                    "task_description": "An easy task"
                },
                "metadata": {"difficulty": "easy", "category": "math"}
            },
            {
                "task": {
                    "time": "2024-01-01T11:00:00", 
                    "main_function_name": "hard_task",
                    "task_description": "A hard task"
                },
                "metadata": {"difficulty": "hard", "category": "string"}
            }
        ]
        
        for item in tasks_with_metadata:
            serialized_task = json.dumps(item["task"])
            tags = {"host": "test_host", "step_id": 1, "task_type": "learnt"}
            # Add metadata to tags
            tags.update(item["metadata"])
            real_config.add_learnt_task(serialized_task, tags)
        
        time.sleep(1)
        
        # Test with filter - this would require extending the few-shot tag format to support filters
        # For now, test basic functionality
        context = {"test": "context"}
        fb_cfg = {
            "use": "collect_feedback",
            "prompt": "Examples:\n\nfew_shots:{\"sources\": \"learnt\", \"num\": 2, \"format\": \"json\"}\n\nComplete task."
        }
        
        result = dynamic_manager_with_real_config._render_feedback_text(context, fb_cfg)
        
        # Should include both tasks since we're not filtering
        assert "few_shots:" not in result
        assert ("easy_task" in result or "hard_task" in result)  # At least one should be present