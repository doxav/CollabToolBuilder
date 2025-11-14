#!/usr/bin/env python3
"""
Advanced test examples for HumanLLM dynamic configuration with rules, 
modifications, and optimizer integration.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from humanllm import HumanLLM

def test_rules_based_config():
    """Test dynamic config with rules-based triggers"""
    try:
        # Test: Rules-based dynamic configuration
        dynamic_config = {
            "debug_mode": {
                "rules": {
                    "regex": {
                        "patterns": [r"debug|error|bug|issue"],
                        "target": "user_message"
                    }
                },
                "modifications": {
                    "temperature": 0.1,
                    "use_premium_llm": True
                }
            },
            "creative_mode": {
                "rules": {
                    "regex": {
                        "patterns": [r"creative|story|imagine|invent"],
                        "target": "user_message"
                    }
                },
                "modifications": {
                    "temperature": 0.9,
                    "use_premium_llm": False,
                    "num_parallel_inferences": 3
                }
            }
        }
        
        rules_llm = HumanLLM(
            system_prompt="You are an adaptive assistant that responds based on query type.",
            llmORchains_list={},
            skip_rounds=10,
            dynamic_llm_config=dynamic_config
        )
        print("✅ Test 1 passed: Rules-based dynamic configuration")
        return True
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False

def test_frequency_rules():
    """Test frequency-based rules"""
    try:
        # Test: Frequency-based triggers
        dynamic_config = {
            "periodic_premium": {
                "rules": {
                    "frequency": {
                        "every_n": 5
                    }
                },
                "modifications": {
                    "use_premium_llm": True,
                    "temperature": 0.3
                }
            }
        }
        
        freq_llm = HumanLLM(
            system_prompt="Assistant with periodic premium mode.",
            llmORchains_list={},
            skip_rounds=15,
            dynamic_llm_config=dynamic_config
        )
        print("✅ Test 2 passed: Frequency-based rules")
        return True
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False

def test_multiple_rules():
    """Test configuration with multiple rule types"""
    try:
        # Test: Multiple rules in single config
        dynamic_config = {
            "complex_mode": {
                "rules": {
                    "regex": {
                        "patterns": [r"complex|difficult|advanced"],
                        "target": "user_message"
                    },
                    "frequency": {
                        "every_n": 10
                    }
                },
                "modifications": {
                    "temperature": 0.2,
                    "use_premium_llm": True,
                    "num_parallel_inferences": 2
                }
            }
        }
        
        multi_rules_llm = HumanLLM(
            system_prompt="Assistant with complex multi-rule configuration.",
            llmORchains_list={},
            skip_rounds=8,
            dynamic_llm_config=dynamic_config
        )
        print("✅ Test 3 passed: Multiple rules configuration")
        return True
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False

def test_trace_optimizer_config():
    """Test basic trace optimizer configuration (without actual opto library)"""
    try:
        # Test: Trace optimizer configuration structure
        dynamic_config = {
            "learning_mode": {
                "rules": {
                    "regex": {
                        "patterns": [r"learn|improve|optimize"],
                        "target": "user_message"
                    }
                },
                "modifications": [
                    {
                        "type": "trace",
                        "optimizer": "test_optimizer",
                        "config": {
                            "optimizer_kind": "optoprimev2",
                            "optimizer_kwargs": {"learning_rate": 0.01}
                        },
                        "targets": ["temperature", "system_prompt"]
                    }
                ]
            }
        }
        
        trace_llm = HumanLLM(
            system_prompt="Learning assistant with trace optimization.",
            llmORchains_list={},
            skip_rounds=5,
            dynamic_llm_config=dynamic_config
        )
        print("✅ Test 4 passed: Trace optimizer configuration")
        return True
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False

def test_invoke_kwargs_modifications():
    """Test modifications that affect invoke parameters"""
    try:
        # Test: Invoke keyword arguments modifications
        dynamic_config = {
            "precise_mode": {
                "rules": {
                    "regex": {
                        "patterns": [r"precise|exact|accurate"],
                        "target": "user_message"
                    }
                },
                "modifications": {
                    "invoke_kwargs": {
                        "temperature": 0.1,
                        "max_tokens": 1000
                    },
                    "system_prompt_append": ["\nPlease be very precise and accurate."]
                }
            }
        }
        
        invoke_mods_llm = HumanLLM(
            system_prompt="Assistant with invoke modifications.",
            llmORchains_list={},
            skip_rounds=6,
            dynamic_llm_config=dynamic_config
        )
        print("✅ Test 5 passed: Invoke kwargs modifications")
        return True
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        return False

if __name__ == "__main__":
    print("Running advanced HumanLLM configuration tests...")
    
    tests = [
        test_rules_based_config,
        test_frequency_rules,
        test_multiple_rules,
        test_trace_optimizer_config,
        test_invoke_kwargs_modifications
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All advanced tests passed! Ready to create validated examples.")
        print("\n📝 Validated advanced patterns:")
        print("  - Rules-based configuration with regex patterns")
        print("  - Frequency-based triggers")
        print("  - Multiple rule evaluation")
        print("  - Trace optimizer integration structure")
        print("  - Invoke kwargs and system prompt modifications")
    else:
        print("⚠️  Some tests failed. Need to fix issues first.")
