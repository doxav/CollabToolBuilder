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