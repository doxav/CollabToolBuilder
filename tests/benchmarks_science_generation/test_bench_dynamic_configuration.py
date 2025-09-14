import pytest
import json
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from sklearn import metrics

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import benchmark
import config


class TestStopException(Exception):
    """Exception raised to stop test execution gracefully (valid test)"""
    pass


class TestDynamicConfiguration:
    """Test suite for dynamic configuration of HumanLLM"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        os.environ["OPENAI"] = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
        os.environ["ENCODER_API_TYPE"] = "openai"
        os.environ["LITELLM_CACHE"] = "disabled"
        os.environ["SearXNG"] = "http://127.0.0.1:8080"
        os.environ["SEARXNG_API_KEY"] = ""
        
        cls.TOPIC = "Taylor Hawkins"
        cls.PROMPT = "Write a brief article about"
        cls.PROMPT_TOPIC = f"{cls.PROMPT} {cls.TOPIC}"
        
    def create_config(self, agent_name: str, dynamic_config: Dict) -> Dict:
        """Create a test configuration"""
        return {
            "agent_name": agent_name,
            "dynamic_llm_config": dynamic_config,
            "automation": True,
        }

    
    def callback(self, human_llm_list, article, step, article_change, metrics):
        """
        Enhanced test callback that stops execution based on quota and step limits.
        """
        human_llms_with_config = [llm for llm in human_llm_list if len(llm.dynamic_llm_config) != 0]
        if not human_llms_with_config:
            return
        human_llm = human_llms_with_config[0]

        # Check quota status if human_llm has usage_tracker
        if hasattr(human_llm, 'usage_tracker') and human_llm.usage_tracker:
            quota_reached = False
            quota_details = []
            
            for help_type, usage in human_llm.usage_tracker.usage.items():
                if help_type in human_llm.usage_tracker.quotas:
                    quota = human_llm.usage_tracker.quotas[help_type]
                    
                    # Check if quota is reached
                    if 'max_count' in quota and usage['count'] >= quota['max_count']:
                        quota_reached = True
                        quota_details.append(f"{help_type}: {usage['count']}/{quota['max_count']} (count)")
                    
                    if 'max_cost' in quota and usage['cost'] >= quota['max_cost']:
                        quota_reached = True
                        quota_details.append(f"{help_type}: {usage['cost']}/{quota['max_cost']} (cost)")
                    
                    if 'max_tokens' in quota and usage['tokens'] >= quota['max_tokens']:
                        quota_reached = True
                        quota_details.append(f"{help_type}: {usage['tokens']}/{quota['max_tokens']} (tokens)")
            
            if quota_reached:
                print(f"Quota reached: {', '.join(quota_details)}. Stopping test gracefully.")
                raise TestStopException(f"Quota reached: {', '.join(quota_details)}")

        
    def run_writehere_test(self, config_payload: Dict, test_name: str,max_steps: int = 10): 
        """Run a single WriteHERE test with given configuration"""
        config_dict = self.create_config("WriteHERE", config_payload)
        try:
            report, turns, metrics = benchmark.run_writehere_human_llm(
                self.TOPIC,
                prompt=self.PROMPT_TOPIC,
                intent="write an article on the topic",
                writehere_model="gpt-4o-mini",
                engine_backend="SearXNG",
                human_llm_parameters=config_dict,
                callback=self.callback,
                max_steps= max_steps,
            )
            return True, f"Test {test_name} passed - completed naturally"
        except benchmark.StopCurrentIteration as e:
            return True, f"Test {test_name} passed - stopped gracefully: {str(e)}"
        except TestStopException as e:
            return True, f"Test {test_name} passed - stopped gracefully: {str(e)}"
        except Exception as e:
            return False, f"Test {test_name} failed with unexpected error: {str(e)}"
        
    def run_storm_test(self, config_payload: Dict, test_name: str, max_steps: int = 10,agent_name: str = "article_gen_lm"):
        """Run a single STORM test with given configuration"""
        dynamic_config = [self.create_config(agent_name, config_payload)]
        try:
            report, turns , metrics = benchmark.run_storm_human_llm(
                self.TOPIC,
                max_conv_turn=1,
                search_top_k=3,
                retrieve_top_k=3,
                max_steps=max_steps,
                callback=self.callback,
                human_llm_parameters=dynamic_config,

            )
            return True, f"Test {test_name} passed - completed naturally"
        except TestStopException as e:
            return True, f"Test {test_name} passed - stopped gracefully: {str(e)}"
        except Exception as e:
            return False, f"Test {test_name} failed with unexpected error: {str(e)}"

    def run_costorm_test(self, config_payload: Dict, test_name: str, max_steps: int = 10):
        """Run a single CoSTORM test with given configuration"""
        config_dict = self.create_config("question_answering_lm", config_payload)
        try:
            report, turns , metrics = benchmark.run_costorm_human_llm(
                self.TOPIC,
                max_conv_turn=1,
                search_top_k=3,
                retrieve_top_k=3,
                max_steps=max_steps,
                callback=self.callback,
                human_llm_parameters=config_dict,
            )
            return True, f"Test {test_name} passed - completed naturally"
        except TestStopException as e:
            return True, f"Test {test_name} passed - stopped gracefully: {str(e)}"
        except Exception as e:
            return False, f"Test {test_name} failed with unexpected error: {str(e)}"

    # =============================================================================
    # RULE TESTS (SIMPLE - NO MODIFICATIONS)
    # =============================================================================
    

    @pytest.mark.parametrize("test_name, config_payload", [
        # RULE TESTS
        ("regex_rule", {
            "regex_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "regex": {
                        "patterns": ["Taylor", "Hawkins"],
                        "target": "user_message"
                    }
                }
            }
        }),
        ("confidence_rule", {
            "confidence_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "confidence": {
                        "threshold": 0.5,
                        "operator": "<"
                    }
                }
            }
        }),
        ("frequency_rule", {
            "frequency_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {
                        "every_n": 2
                    }
                }
            }
        }),
        ("complexity_rule", {
            "complexity_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "complexity": {
                        "threshold": 0.3
                    }
                }
            }
        }),
        ("similarity_rule", {
            "similarity_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "similarity": {
                        "threshold": 0.7,
                        "min_similar": 2
                    }
                }
            }
        }),
        ("history_rule", {
            "history_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "history": {
                        "lookback": 5,
                        "success_threshold": 0.8
                    }
                }
            }
        }),
        ("composite_rule_and", {
            "composite_and_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "composite": {
                        "operator": "AND",
                        "rules": {
                            "regex": {
                                "patterns": ["Taylor"],
                                "target": "user_message"
                            },
                            "complexity": {
                                "threshold": 0.2
                            }
                        }
                    }
                }
            }
        }),
        ("composite_rule_or", {
            "composite_or_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "composite": {
                        "operator": "OR",
                        "rules": {
                            "frequency": {
                                "every_n": 2
                            },
                            "complexity": {
                                "threshold": 0.8
                            }
                        }
                    }
                }
            }
        }),
        # MODIFICATIONS
        ("use_premium_llm", {
            "premium_llm_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "use_premium_llm": True
                }
            }
        }),
        ("parallel_inferences", {
            "parallel_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "num_parallel_inferences": 3
                }
            }
        }),
        # GENERATION TECHNIQUES
        *[
            (name, {
                f"{name}_test": {
                    "phase": "pre_inference",
                    "quota": {"max_count": 3},
                    "rules": {
                        "frequency": {"every_n": 1}
                    },
                    "modifications": {
                        "num_parallel_inferences": 2,
                        "generation_technique": name
                    }
                }
            }) for name in [
                "temperature_variation", "self_refinement", "multi_experts",
                "mixture_of_agents_generation", "iterative_alternatives"
            ]
        ],
        # SELECTION TECHNIQUES
        *[
            (name, {
                f"{name}_test": {
                    "phase": "pre_inference",
                    "quota": {"max_count": 3},
                    "rules": {
                        "frequency": {"every_n": 1}
                    },
                    "modifications": {
                        "num_parallel_inferences": 2,
                        "selection_technique": name.split("_")[-1]
                    }
                }
            }) for name in [
                "selection_best_of_n", "selection_concat", "selection_majority",
                "selection_moa", "selection_last"
            ]
        ],
        # FEEDBACK TYPES
        ("annotations_fix", {
            "annotations_fix_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_annotations_feedback": {
                        "annotation_types": "FIX"
                    },
                    "apply_feedback": True
                }
            }
        }),
        ("annotations_improve", {
            "annotations_improve_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_annotations_feedback": {
                        "annotation_types": "IMPROVE"
                    },
                    "apply_feedback": True
                }
            }
        }),
        ("annotations_delete", {
            "annotations_delete_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_annotations_feedback": {
                        "annotation_types": "DELETE"
                    },
                    "apply_feedback": True
                }
            }
        }),
        ("annotations_fulltext_all", {
            "annotations_fulltext_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_annotations_feedback": {
                        "annotation_types": "FIX"
                    },
                    "apply_feedback": True,
                    "apply_feedback_params": {
                        "instruction_processing_approach": "FULLTEXT_ALL"
                    }
                }
            }
        }),
        ("annotations_fulltext_each", {
            "annotations_fulltext_each_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_annotations_feedback": {
                        "annotation_types": "FIX"
                    },
                    "apply_feedback": True,
                    "apply_feedback_params": {
                        "instruction_processing_approach": "FULLTEXT_EACH"
                    }
                }
            }
        }),
        ("annotations_all", {
            "annotations_all_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_annotations_feedback": {
                        "annotation_types": "FIX"
                    },
                    "apply_feedback": True,
                    "apply_feedback_params": {
                        "instruction_processing_approach": "ANNOTATIONS_ALL"
                    }
                }
            }
        }),
        ("annotations_each", {
            "annotations_all_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_annotations_feedback": {
                        "annotation_types": "FIX"
                    },
                    "apply_feedback": True,
                    "apply_feedback_params": {
                        "instruction_processing_approach": "ANNOTATIONS_EACH"
                    }
                }
            }
        }),
        ("instructions_feedback", {
            "instructions_basic_test": {
                "phase": "post_inference",
                "quota": {"max_count": 3},
                "rules": {
                    "frequency": {"every_n": 1}
                },
                "modifications": {
                    "generate_instructions_feedback": {},
                    "apply_feedback": True
                }
            }
        }),
        # QUOTA TESTS
        ("quota_max_count", {
            "quota_count_test": {
                "phase": "pre_inference",
                "quota": {"max_count": 1},
                "rules": {
                    "frequency": {"every_n": 1}
                }
            }
        }),
        ("quota_max_cost", {
            "quota_cost_test": {
                "phase": "pre_inference",
                "quota": {"max_cost": 0.01},
                "rules": {
                    "frequency": {"every_n": 1}
                }
            }
        }),
        ("quota_max_tokens", {
            "quota_tokens_test": {
                "phase": "pre_inference",
                "quota": {"max_tokens": 100},
                "rules": {
                    "frequency": {"every_n": 1}
                }
            }
        })
    ])
    def test_writehere_rules(self, test_name, config_payload, mode="writehere"):
        #success, message = self.run_storm_test(config_payload, test_name, max_steps=1)
        if mode == "storm":
            success, message = self.run_storm_test(config_payload, test_name, max_steps=1)
        elif mode == "costorm":
            success, message = self.run_costorm_test(config_payload, test_name, max_steps=1)
        elif mode == "writehere":
            success, message = self.run_writehere_test(config_payload, test_name, max_steps=1)
        #elif mode == "aiscientistv2":
        #    success, message = self.run_aiscientistv2_test(config_payload, test_name, max_steps=1)
        #elif mode == "agentlaboratory":
        #    success, message = self.run_agentlaboratory_test(config_payload, test_name, max_steps=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        assert success, message


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
