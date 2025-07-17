# test_dynamic_llm_config.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from utils.human_llm import HumanLLM, DynamicConfigManager, HelpUsageTracker
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from config import MODELS_CONFIG_LIST

class TestHelpUsageTracker:
    def test_quota_tracking(self):
        tracker = HelpUsageTracker()
        
        # Set quota
        tracker.set_quota('premium_help', {'max_count': 5, 'max_cost': 10.0})
        
        # Should allow usage initially
        assert tracker.can_use('premium_help') == True
        
        # Record usage up to limit
        for i in range(5):
            tracker.record_usage('premium_help', cost=1.0, tokens=100)
            
        # Should block after limit
        assert tracker.can_use('premium_help') == False
        
    def test_no_quota_always_allows(self):
        tracker = HelpUsageTracker()
        
        # No quota set means unlimited
        for i in range(100):
            assert tracker.can_use('unlimited_help') == True
            tracker.record_usage('unlimited_help')

class TestDynamicConfigManager:
    def setup_method(self):
        self.usage_tracker = HelpUsageTracker()
        self.default_config = {
            'num_parallel_inferences': 1,
            'temperature_min': 0.0,
            'temperature_max': 0.2
        }
        
    def test_regex_rule_evaluation(self):
        config = {
            'complex_query': {
                'phase': 'pre_inference',
                'rules': {
                    'regex': {
                        'patterns': [r'complex.*question', r'difficult.*problem'],
                        'target': 'user_message'
                    }
                },
                'modifications': {
                    'num_parallel_inferences': 3,
                    'temperature_max': 0.8
                }
            }
        }
        
        manager = DynamicConfigManager(config, self.usage_tracker, self.default_config)
        
        # Should trigger on matching pattern
        context = {'user_message': 'This is a complex question about AI'}
        mods = manager.evaluate_triggers(context, phase='pre_inference')
        assert mods == {'num_parallel_inferences': 3, 'temperature_max': 0.8}
        
        # Should not trigger on non-matching
        context = {'user_message': 'Simple query'}
        mods = manager.evaluate_triggers(context, phase='pre_inference')
        assert mods == {}
        
    def test_confidence_rule(self):
        config = {
            'low_confidence': {
                'phase': 'pre_inference',
                'rules': {
                    'confidence': {
                        'threshold': 0.7,
                        'operator': '<'
                    }
                },
                'modifications': {
                    'activate_human_intervention': True
                }
            }
        }
        
        manager = DynamicConfigManager(config, self.usage_tracker, self.default_config)
        
        # Low confidence should trigger
        context = {'confidence_score': 0.5}
        mods = manager.evaluate_triggers(context)
        assert mods['activate_human_intervention'] == True
        
        # High confidence should not trigger
        context = {'confidence_score': 0.9}
        mods = manager.evaluate_triggers(context)
        assert mods == {}
        
    def test_divergence_rule(self):
        config = {
            'high_divergence': {
                'phase': 'post_inference',
                'rules': {
                    'divergence': {
                        'threshold': 0.5
                    }
                },
                'modifications': {
                    'generate_annotations_feedback': {
                        'annotation_types': 'FIX, IMPROVE',
                        'annotation_number': 10
                    },
                    'apply_feedback': True
                }
            }
        }
        
        manager = DynamicConfigManager(config, self.usage_tracker, self.default_config)
        
        # Similar outputs should not trigger
        outputs = [
            Mock(content="The answer is 42"),
            Mock(content="The answer is 42, clearly")
        ]
        context = {'llm_outputs': outputs}
        mods = manager.evaluate_triggers(context, phase='post_inference')
        assert mods == {}
        
        # Divergent outputs should trigger
        outputs = [
            Mock(content="The answer is 42"),
            Mock(content="I cannot determine the answer")
        ]
        context = {'llm_outputs': outputs}
        mods = manager.evaluate_triggers(context, phase='post_inference')
        assert 'generate_annotations_feedback' in mods
        assert mods['apply_feedback'] == True
        
    def test_composite_rule(self):
        config = {
            'complex_and_uncertain': {
                'phase': 'pre_inference',
                'rules': {
                    'composite': {
                        'operator': 'AND',
                        'rules': {
                            'complexity': {'threshold': 0.5},
                            'regex': {'patterns': [r'\?'], 'target': 'user_message'}
                        }
                    }
                },
                'modifications': {
                    'use_premium_llm': True,
                    'num_parallel_inferences': 5
                }
            }
        }
        
        manager = DynamicConfigManager(config, self.usage_tracker, self.default_config)
        
        # Should trigger when both conditions met
        context = {'user_message': 'What is the complex relationship between quantum mechanics and consciousness?'}
        mods = manager.evaluate_triggers(context)
        assert mods.get('use_premium_llm') == True
        assert mods.get('num_parallel_inferences') == 5

        # Should not trigger with only one condition
        context = {'user_message': 'What is 2+2'}  # Has ? but not complex
        mods = manager.evaluate_triggers(context)
        assert mods == {}

class TestHumanLLMDynamicConfig:
    def setup_method(self):
        self.dynamic_config = {
            'uncertainty_help': {
                'phase': 'post_inference',
                'quota': {'max_count': 10},
                'rules': {
                    'divergence': {'threshold': 0.4}
                },
                'modifications': {
                    'generate_instructions_feedback': {
                        'num_candidates': 3
                    },
                    'selection_technique': 'majority'
                }
            },
            'complex_help': {
                'phase': 'pre_inference',
                'quota': {'max_cost': 5.0},
                'rules': {
                    'complexity': {'threshold': 0.6}
                },
                'modifications': {
                    'num_parallel_inferences': 5,
                    'temperature_max': 0.9,
                    'generation_technique': 'multi_experts'
                }
            }
        }
        
    def test_pre_inference_modifications(self):
        llm = HumanLLM(
            default_llm_choice="test",
            llmORchains_list={'test': Mock()},
            dynamic_llm_config=self.dynamic_config
        )
        
        # Mock complex user message
        context = {
            'user_message': ' '.join(['complex'] * 100),  # Long complex message
            'phase': 'pre_inference'
        }
        
        # Store original values
        orig_parallel = llm.num_parallel_inferences
        orig_temp = llm.temperature_max
        
        # Apply dynamic config
        llm._apply_dynamic_config(context, phase='pre_inference')
        
        # Check modifications applied
        assert llm.num_parallel_inferences == 5
        assert llm.temperature_max == 0.9
        assert context.get('dynamic_modifications') is not None
        
    def test_post_inference_feedback_generation(self):
        llm = HumanLLM(
            default_llm_choice="test",
            llmORchains_list={'test': Mock()},
            dynamic_llm_config=self.dynamic_config
        )
        
        # Mock divergent outputs
        outputs = [
            AIMessage(content="Solution A using method 1"),
            AIMessage(content="Completely different approach B")
        ]
        
        context = {
            'llm_outputs': outputs,
            'phase': 'post_inference'
        }
        
        # Mock feedback methods
        llm.generate_instructions_feedback = Mock(return_value={
            'suggestions': 'Improve clarity',
            'improvement_prompt': 'test'
        })
        llm.apply_feedback = Mock(return_value="Improved content")
        
        # Apply dynamic config
        llm._apply_dynamic_config(context, phase='post_inference')
        
        # Check feedback was generated
        assert llm.generate_instructions_feedback.called
        assert llm.selection_technique == 'majority'
        
    def test_quota_enforcement(self):
        llm = HumanLLM(
            default_llm_choice="test",
            llmORchains_list={'test': Mock()},
            dynamic_llm_config=self.dynamic_config
        )
        
        # Exhaust quota
        for i in range(10):
            llm.usage_tracker.record_usage('uncertainty_help')
            
        # Should not apply modifications after quota exhausted
        context = {
            'llm_outputs': [Mock(content="A"), Mock(content="B")],
            'phase': 'post_inference'
        }
        
        llm._apply_dynamic_config(context, phase='post_inference')
        
        # No modifications should be applied
        assert context.get('dynamic_modifications', {}) == {}
        
    def test_human_intervention_activation(self):
        config = {
            'force_human': {
                'phase': 'pre_inference',
                'rules': {
                    'regex': {'patterns': [r'HELP'], 'target': 'user_message'}
                },
                'modifications': {
                    'activate_human_intervention': True
                }
            }
        }
        
        llm = HumanLLM(
            default_llm_choice="test",
            llmORchains_list={'test': Mock()},
            dynamic_llm_config=config,
            skip_rounds=5  # Start with automation
        )
        
        context = {'user_message': 'I need HELP with this', 'phase': 'pre_inference'}
        
        # Should activate human intervention
        assert llm.skip_rounds == 5
        llm._apply_dynamic_config(context, phase='pre_inference')
        assert llm.skip_rounds == 0
        
    def test_full_workflow_integration(self):
       """Test a complete workflow with multiple phases and modifications"""
       config = {
           'adaptive_generation': {
               'phase': 'pre_inference',
               'rules': {
                   'frequency': {'every_n': 3}
               },
               'modifications': {
                   'num_parallel_inferences': 3,
                   'generation_technique': 'mixture_of_agents'
               }
           },
           'adaptive_feedback': {
               'phase': 'post_inference',
               'rules': {
                   'divergence': {'threshold': 0.6}
               },
               'modifications': {
                   'generate_annotations_feedback': {
                       'annotation_types': 'FIX, IMPROVE, DELETE',
                       'annotation_number': 5
                   },
                   'apply_feedback': True,
                   'reselect_best': True,
                   'reselection_technique': 'best_of_n'
               }
           }
       }
       
       # Create HumanLLM with mocked components
       with patch('utils.human_llm.HumanLLMConfig') as mock_config:
           llm = HumanLLM(
               default_llm_choice="test",
               llmORchains_list={'test': Mock()},
               dynamic_llm_config=config
           )
           
           # Track calls through the workflow
           llm.generate_annotations_feedback = Mock(return_value={
               'annotations': '[FIX]This needs improvement[/FIX]'
           })
           llm.apply_feedback = Mock(return_value="Improved content")
           llm.select_candidate = Mock(return_value=[AIMessage(content="Best output")])
           
           # Simulate 3rd call (frequency trigger)
           for i in range(2):
               llm.dynamic_mgr._call_history.append({
                   'context': {'user_message': f'Query {i}'},
                   'phase': 'pre_inference',
                   'timestamp': datetime.now()
               })
           
           # Pre-inference context
           pre_context = {
               'user_message': 'Query 3',
               'phase': 'pre_inference'
           }
           
           # Apply pre-inference config
           llm._apply_dynamic_config(pre_context, phase='pre_inference')
           
           # Verify pre-inference modifications
           assert llm.num_parallel_inferences == 3
           assert llm.generation_technique == 'mixture_of_agents'
           
           # Post-inference context with divergent outputs
           post_context = {
               'llm_outputs': [
                   AIMessage(content="Answer A"),
                   AIMessage(content="Completely different B"),
                   AIMessage(content="Yet another approach C")
               ],
               'phase': 'post_inference'
           }
           
           # Apply post-inference config
           llm._apply_dynamic_config(post_context, phase='post_inference')
           
           # Verify feedback was generated and applied
           assert llm.generate_annotations_feedback.called
           assert llm.apply_feedback.called
           assert llm.select_candidate.called
           
    def test_error_handling(self):
       """Test error handling in dynamic configuration"""
       config = {
           'bad_rule': {
               'phase': 'pre_inference',
               'rules': {
                   'nonexistent_rule_type': {'some': 'config'}
               },
               'modifications': {
                   'num_parallel_inferences': 5
               }
           }
       }
       
       llm = HumanLLM(
           default_llm_choice="test",
           llmORchains_list={'test': Mock()},
           dynamic_llm_config=config
       )
       
       # Should handle unknown rule type gracefully
       context = {'user_message': 'Test', 'phase': 'pre_inference'}
       llm._apply_dynamic_config(context, phase='pre_inference')
       
       # No modifications should be applied due to unknown rule
       assert llm.num_parallel_inferences == 1  # Default value
       
    def test_history_based_rules(self):
       """Test rules that depend on historical performance"""
       config = {
           'performance_boost': {
               'phase': 'pre_inference',
               'rules': {
                   'history': {
                       'lookback': 5,
                       'success_threshold': 0.5
                   }
               },
               'modifications': {
                   'use_premium_llm': True
               }
           }
       }
       
       llm = HumanLLM(
           default_llm_choice="test",
           llmORchains_list={'test': Mock()},
           dynamic_llm_config=config
       )
       
       # Add history with poor performance
       for i in range(5):
           llm.dynamic_mgr._call_history.append({
               'context': {
                   'validation_result': 'failed' if i < 3 else 'success'
               },
               'phase': 'pre_inference',
               'timestamp': datetime.now()
           })
       
       context = {'user_message': 'New query', 'phase': 'pre_inference'}
       llm._apply_dynamic_config(context, phase='pre_inference')
       
       # Should activate premium LLM due to low success rate (2/5 = 0.4)
       assert context.get('use_premium_llm') == True
       
    def test_feedback_application_with_annotations(self):
       """Test the full feedback application flow with annotations"""
       config = {
           'annotation_feedback': {
               'phase': 'post_inference',
               'rules': {
                   'regex': {'patterns': ['improve'], 'target': 'user_message'}
               },
               'modifications': {
                   'generate_annotations_feedback': {
                       'annotation_types': 'FIX, IMPROVE',
                       'annotation_number': 3,
                       'generation_technique': 'temperature_variation',
                       'num_candidates': 2
                   },
                   'apply_feedback': True,
                   'apply_feedback_params': {
                       'annotation_format': 'latex-inline',
                       'instruction_processing_approach': 'ANNOTATIONS_ALL'
                   }
               }
           }
       }

       llm = HumanLLM(
           default_llm_choice="test",
           #llmORchains_list={'test': Mock()},
           llmORchains_list={'test': ChatOpenAI(model_name=MODELS_CONFIG_LIST.getattr("premium_llm", "gpt-4.1-nano") if MODELS_CONFIG_LIST else "gpt-4.1-nano"), 'premium_llm': ChatOpenAI(model_name=MODELS_CONFIG_LIST.getattr("premium_llm", "gpt-4.1-nano") if MODELS_CONFIG_LIST else "gpt-4.1-nano")},
           #llmORchains_list={'test': mock_test_llm, 'premium_llm': mock_test_llm},
           temperature_min= 0.0,
           dynamic_llm_config=config
       )
       
       # Mock the feedback generation
       llm.generate_annotations_feedback_fn = Mock(return_value={
           'annotations': '\\FIX{error here}\\IMPROVE{make clearer}',
           'annotation_prompt': 'test prompt'
       })
       
       # Mock the feedback application
       llm.apply_feedback_fn = Mock(return_value="Fixed and improved content")
       
       outputs = [AIMessage(content="Original content with error here and unclear parts")]
       context = {
           'user_message': 'Please improve this response',
           'llm_outputs': outputs,
           'phase': 'post_inference'
       }
       
       # Apply configuration
       llm._apply_dynamic_config(context, phase='post_inference')
       
       # Verify the flow
       assert llm.generate_annotations_feedback_fn.called
       call_args = llm.generate_annotations_feedback_fn.call_args
       assert call_args[1]['annotation_types'] == 'FIX, IMPROVE'
       assert call_args[1]['num_candidates'] == 2
       
       assert llm.apply_feedback_fn.called
       apply_args = llm.apply_feedback_fn.call_args
       assert apply_args[1]['annotation_format'] == 'latex-inline'
       
       # Output should be updated
       assert outputs[0].content == "Fixed and improved content"
       
    def test_multiple_help_types_interaction(self):
       """Test interaction between multiple help types"""
       config = {
           'basic_help': {
               'phase': 'pre_inference',
               'rules': {
                   'regex': {'patterns': ['help'], 'target': 'user_message'}
               },
               'modifications': {
                   'temperature_max': 0.5
               }
           },
           'advanced_help': {
               'phase': 'pre_inference',
               'rules': {
                   'regex': {'patterns': ['complex help'], 'target': 'user_message'}
               },
               'modifications': {
                   'temperature_max': 0.9,
                   'num_parallel_inferences': 5,
                   'use_premium_llm': True
               }
           }
       }
       
       llm = HumanLLM(
           default_llm_choice="test",
           llmORchains_list={'test': Mock()},
           dynamic_llm_config=config
       )
       
       # Test with basic help - only first rule matches
       context = {'user_message': 'I need help', 'phase': 'pre_inference'}
       llm._apply_dynamic_config(context, phase='pre_inference')
       assert llm.temperature_max == 0.5
       assert llm.num_parallel_inferences == 1  # Default
       
       # Reset
       llm.temperature_max = 0.0
       llm.num_parallel_inferences = 1
       
       # Test with advanced help - both rules match, later one wins
       context = {'user_message': 'I need complex help', 'phase': 'pre_inference'}
       llm._apply_dynamic_config(context, phase='pre_inference')
       assert llm.temperature_max == 0.9
       assert llm.num_parallel_inferences == 5
       assert context.get('use_premium_llm') == True
       
    @pytest.mark.parametrize("rule_config,context,should_trigger", [
       # Regex rule tests
       (
           {'regex': {'patterns': [r'\d+'], 'target': 'user_message'}},
           {'user_message': 'Calculate 123 + 456'},
           True
       ),
       (
           {'regex': {'patterns': [r'\d+'], 'target': 'user_message'}},
           {'user_message': 'No numbers here'},
           False
       ),
       # Confidence rule tests
       (
           {'confidence': {'threshold': 0.3, 'operator': '<'}},
           {'confidence_score': 0.2},
           True
       ),
       (
           {'confidence': {'threshold': 0.3, 'operator': '>'}},
           {'confidence_score': 0.5},
           True
       ),
       # Complexity rule tests
       (
           {'complexity': {'threshold': 0.5}},
           {'user_message': 'What is ' + ' '.join(['the'] * 50) + ' meaning?'},
           True
       ),
       (
           {'complexity': {'threshold': 0.5}},
           {'user_message': 'Hi'},
           False
       ),
   ])
    def test_individual_rule_evaluation(self, rule_config, context, should_trigger):
       """Parameterized tests for individual rule evaluations"""
       config = {
           'test_help': {
               'phase': 'pre_inference',
               'rules': rule_config,
               'modifications': {'test': True}
           }
       }
       
       manager = DynamicConfigManager(config, HelpUsageTracker(), {})
       mods = manager.evaluate_triggers(context, phase='pre_inference')
       
       if should_trigger:
           assert 'test' in mods
       else:
           assert mods == {}
           
    def test_usage_tracking_integration(self):
       """Test that usage is properly tracked when help is applied"""
       config = {
           'premium_feature': {
               'phase': 'pre_inference',
               'quota': {'max_count': 3},
               'rules': {
                   'regex': {'patterns': ['premium'], 'target': 'user_message'}
               },
               'modifications': {
                   'use_premium_llm': True
               }
           }
       }
       
       llm = HumanLLM(
           default_llm_choice="test",
           llmORchains_list={'test': Mock()},
           dynamic_llm_config=config
       )
       
       # Use the feature 3 times
       for i in range(3):
           context = {'user_message': 'Use premium feature', 'phase': 'pre_inference'}
           llm._apply_dynamic_config(context, phase='pre_inference')
           
       # Check usage was tracked
       usage = llm.usage_tracker.usage['premium_feature']
       assert usage['count'] == 3
       
       # 4th attempt should be blocked by quota
       context = {'user_message': 'Use premium feature', 'phase': 'pre_inference'}
       llm._apply_dynamic_config(context, phase='pre_inference')
       
       # Usage count should still be 3
       assert llm.usage_tracker.usage['premium_feature']['count'] == 3
       
    def test_phase_specific_rules(self):
       """Test that rules only apply in their specified phase"""
       config = {
           'pre_only': {
               'phase': 'pre_inference',
               'rules': {'regex': {'patterns': ['test']}},
               'modifications': {'temperature_max': 0.9}
           },
           'post_only': {
               'phase': 'post_inference',
               'rules': {'regex': {'patterns': ['test']}},
               'modifications': {'apply_feedback': True}
           }
       }
       
       llm = HumanLLM(
           default_llm_choice="test",
           llmORchains_list={'test': Mock()},
           dynamic_llm_config=config
       )
       
       context = {'user_message': 'test message'}
       
       # Pre-inference should only apply pre_only
       llm._apply_dynamic_config(context, phase='pre_inference')
       assert llm.temperature_max == 0.9
       assert 'apply_feedback' not in context.get('dynamic_modifications', {})
       
       # Reset
       llm.temperature_max = 0.0
       
       # Post-inference should only apply post_only
       context['llm_outputs'] = [AIMessage(content="test")]
       llm._apply_dynamic_config(context, phase='post_inference')
       assert llm.temperature_max == 0.0  # Not modified
       assert context.get('dynamic_modifications', {}).get('apply_feedback') == True


if __name__ == "__main__":
   pytest.main([__file__, "-v"])