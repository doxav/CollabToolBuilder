import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.human_llm import HumanLLM, DynamicConfigManager, HelpUsageTracker


def make_stub_llm(dynamic_config, **kwargs):
    """Create a HumanLLM instance with minimal dependencies for testing"""
    default_kwargs = {
        'llmORchains_list': {'default_llm': MagicMock(), 'premium_llm': MagicMock()},
        'dynamic_llm_config': dynamic_config
    }
    default_kwargs.update(kwargs)
    
    with patch('utils.human_llm.HumanLLMConfig'), \
         patch.object(HumanLLM, 'set_print_color'), \
         patch.object(HumanLLM, 'configure_output_schema'), \
         patch.object(HumanLLM, 'set_default_llmORchain'), \
         patch.object(HumanLLM, 'set_premium_llmORchain'):
        llm = HumanLLM(**default_kwargs)
    return llm


class TestDynamicLLMConfigEssential:
    """Essential backward compatibility tests for dynamic config"""
    
    def test_basic_initialization(self):
        """Test that dynamic config system initializes correctly"""
        config = {}
        llm = make_stub_llm(config)
        
        assert hasattr(llm, 'dynamic_mgr')
        assert hasattr(llm, 'usage_tracker')
        assert llm.dynamic_llm_config == config
    
    def test_frequency_rule(self):
        """Test frequency-based rule triggering"""
        config = {
            'help_every_3': {
                'phase': 'pre_inference',
                'rules': {
                    'frequency': {'every_n': 3}
                },
                'modifications': {
                    'num_parallel_inferences': 2
                }
            }
        }
        llm = make_stub_llm(config)
        
        # Add 2 calls to history (making next one the 3rd)
        for i in range(2):
            llm.dynamic_mgr._call_history.append({
                'context': {'user_message': f'Test {i}'},
                'phase': 'pre_inference',
                'timestamp': datetime.now()
            })
        
        context = {'user_message': 'Third call'}
        mods = llm.dynamic_mgr.evaluate_triggers(context, phase='pre_inference')
        
        assert 'num_parallel_inferences' in mods
        assert mods['num_parallel_inferences'] == 2
    
    def test_confidence_rule(self):
        """Test confidence/divergence rule evaluation"""
        config = {
            'low_confidence_help': {
                'phase': 'post_inference', 
                'rules': {
                    'confidence': {
                        'method': 'self-consistency',
                        'threshold': 0.8,
                        'consistency_method': 'difflib'
                    }
                },
                'modifications': {
                    'activate_human_intervention': True
                }
            }
        }
        llm = make_stub_llm(config)
        
        # Mock outputs that are very different (low similarity)
        class MockOutput:
            def __init__(self, content):
                self.content = content
                
        outputs = [
            MockOutput("Answer A is completely different"),
            MockOutput("Solution B uses totally different approach"),
        ]
        
        context = {
            'llm_outputs': outputs,
            'phase': 'post_inference'
        }
        
        mods = llm.dynamic_mgr.evaluate_triggers(context, phase='post_inference')
        
        # Should trigger due to low similarity
        assert 'activate_human_intervention' in mods
        assert mods['activate_human_intervention'] is True
    
    def test_modifications_applied_correctly(self):
        """Test that modifications are applied to HumanLLM instance"""
        config = {
            'test_mod': {
                'phase': 'pre_inference',
                'rules': {
                    'frequency': {'every_n': 1}  # Always trigger
                },
                'modifications': {
                    'num_parallel_inferences': 5,
                    'temperature_min': 0.8,
                    'temperature_max': 0.9
                }
            }
        }
        
        llm = make_stub_llm(config)
        original_num = llm.num_parallel_inferences
        original_temp_min = llm.temperature_min
        original_temp_max = llm.temperature_max
        
        context = {'user_message': 'test'}
        mods = llm.dynamic_mgr.evaluate_triggers(context, phase='pre_inference')
        llm._apply_modifications(mods, context, 'pre_inference')
        
        assert llm.num_parallel_inferences == 5
        assert llm.temperature_min == 0.8
        assert llm.temperature_max == 0.9
        
        # Test that original values were tracked
        assert 'original_values' in context
        assert context['original_values']['num_parallel_inferences'] == original_num
    
    def test_no_rules_no_modifications(self):
        """Test that no modifications are applied when rules don't match"""
        config = {
            'strict_rule': {
                'phase': 'pre_inference',
                'rules': {
                    'regex': {
                        'patterns': ['NEVER_MATCHES_ANYTHING_UNIQUE_12345'],
                        'target': 'user_message'
                    }
                },
                'modifications': {
                    'num_parallel_inferences': 999
                }
            }
        }
        
        llm = make_stub_llm(config)
        original_num = llm.num_parallel_inferences
        
        context = {'user_message': 'normal message'}
        mods = llm.dynamic_mgr.evaluate_triggers(context, phase='pre_inference')
        llm._apply_modifications(mods, context, 'pre_inference')
        
        # Should remain unchanged
        assert llm.num_parallel_inferences == original_num
