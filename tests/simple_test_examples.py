#!/usr/bin/env python3
"""
Simple test examples for HumanLLM that focus on basic functionality
without complex mocking or external API calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from humanllm import HumanLLM

def test_basic_initialization():
    """Test basic HumanLLM initialization without LLM calls"""
    try:
        # Test 1: Basic initialization
        basic_llm = HumanLLM(
            system_prompt="You are a helpful assistant.",
            llmORchains_list={},  # Empty dict to avoid API calls
            skip_rounds=10
        )
        print("✅ Test 1 passed: Basic initialization")
        return True
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False

def test_named_llms():
    """Test HumanLLM with named default and premium LLMs"""
    try:
        # Test 2: Named LLMs with dictionary
        from langchain_openai import ChatOpenAI
        
        llms_dict = {
            "default_llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5),
            "premium_llm": ChatOpenAI(model="gpt-4", temperature=0.3)
        }
        
        named_llm = HumanLLM(
            system_prompt="You are a helpful assistant.",
            llmORchains_list=llms_dict,
            default_llmORchain="default_llm",
            premium_llmORchain="premium_llm",
            premium_llm_by_default=False,
            skip_rounds=10
        )
        print("✅ Test 2 passed: Named default and premium LLMs")
        return True
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False

def test_dynamic_llm_config():
    """Test HumanLLM with dynamic configuration"""
    try:
        # Test 3: Dynamic configuration
        dynamic_config = {
            "creative": {
                "temperature": 0.9,
                "use_premium_llm": False,
                "quota": 5
            },
            "analytical": {
                "temperature": 0.2,
                "use_premium_llm": True,
                "num_parallel_inferences": 2
            },
            "default": {
                "temperature": 0.5,
                "use_premium_llm": False
            }
        }
        
        config_llm = HumanLLM(
            system_prompt="You are an adaptive assistant.",
            llmORchains_list={},
            skip_rounds=5,
            dynamic_llm_config=dynamic_config
        )
        print("✅ Test 3 passed: Dynamic configuration")
        return True
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False

def test_premium_by_default():
    """Test premium LLM as default"""
    try:
        # Test 4: Premium LLM by default
        premium_default = HumanLLM(
            system_prompt="You are a premium assistant.",
            llmORchains_list={},
            premium_llm_by_default=True,
            skip_rounds=3
        )
        print("✅ Test 4 passed: Premium LLM by default")
        return True
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False

if __name__ == "__main__":
    print("Running enhanced HumanLLM tests...")
    
    tests = [
        test_basic_initialization,
        test_named_llms,
        test_dynamic_llm_config,
        test_premium_by_default
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to create validated examples.")
        print("\n📝 Validated patterns:")
        print("  - llmORchains_list as dictionary with named LLMs")
        print("  - default_llmORchain and premium_llmORchain parameters")
        print("  - premium_llm_by_default flag") 
        print("  - dynamic_llm_config with help types and configurations")
    else:
        print("⚠️  Some tests failed. Need to fix issues first.")
