import pytest
import time
import difflib
import os
import requests
import random
import re
import sys

from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from tabulate import tabulate

from utils.human_llm import HumanLLM, HumanLLMConfig
from utils.llm_utils import UnifiedVectorDB, UnifiedVectorDBConfig
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem_phrase(phrase):
    return [stemmer.stem(word.lower()) for word in re.findall(r"\w+", phrase)]

class DummyEmbedding:
    """Simple embedding function for testing purposes."""
    
    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_query(self, text: str):
        return [1.0] * self.dim

    def embed_documents(self, texts):
        return [[1.0] * self.dim for _ in texts]

_FULL_HAMLET: str = None
# Cache and scramble snippets deterministically per length
_SCRAMBLED: Dict[int, str] = {}

def fetch_hamlet_text(char_limit: int = None) -> str:
    """Fetch full spoken-text from DraCor and truncate to char_limit."""
    global _FULL_HAMLET
    if _FULL_HAMLET is None:
        resp = requests.get("https://dracor.org/api/v0/corpora/shake/play/hamlet/spoken-text")
        resp.raise_for_status()
        # this endpoint returns the raw play‐text, not JSON
        _FULL_HAMLET = resp.text
    return _FULL_HAMLET[:char_limit] if char_limit else _FULL_HAMLET

def get_scrambled_hamlet(char_limit: int) -> Tuple[str, str]:
    """Return both scrambled and properly formatted reference."""
    if char_limit not in _SCRAMBLED:
        orig = fetch_hamlet_text(char_limit)
        
        # Simpler split: only on actual line breaks
        parts = re.split(r'[\r\n]+', orig)
        parts = [p.strip() for p in parts if p.strip()]  # Remove empty lines
        
        # display the length of orig vs length of parts + len(parts)*1 or 2
        text = f""""
        Original length: {len(orig)} chars
        Total length without line breaks: {sum(len(p) for p in parts)} chars
        Total parts: {len(parts)}
        Total length with line breaks: {sum(len(p) + 1 for p in parts)} chars
        Total length with line breaks + 2: {sum(len(p) + 2 for p in parts)} chars"""
        print(text)
        # Create consistent reference (same formatting as output will have)
        reference_formatted = "\n".join(parts)
        
        # Scramble the parts
        rnd = random.Random(char_limit)
        rnd.shuffle(parts)
        scrambled = "\n".join(parts)
        
        _SCRAMBLED[char_limit] = (scrambled, reference_formatted)
    
    return _SCRAMBLED[char_limit]

# default fallback, if someone still uses the env var
HAMLET_CHAR_LIMIT = int(os.getenv("HAMLET_CHAR_LIMIT", "1000"))

@pytest.fixture(autouse=True)
def reset_human_llmconfig_singleton():
    """Reset the HumanLLMConfig singleton for each test."""
    HumanLLMConfig._HumanLLMConfig__instance = None
    yield


@pytest.fixture
def vectordb_backend(tmp_path):
    """Create a vector database for testing."""
    from utils.llm_utils import CHROMA_DATABASE
    
    config = UnifiedVectorDBConfig(
        embedding_function=DummyEmbedding(dim=16),
        collection_name="test_hamlet",
        persist_directory=str(tmp_path / "vectordb_data"),
        reset_indices=True
    )
    config.db_type = CHROMA_DATABASE
    
    uvdb = UnifiedVectorDB(config=config, check_db=True)
    yield uvdb
    
    # Cleanup
    try:
        all_ids = [doc.id for doc in uvdb.db._collection.get()]
        if all_ids:
            uvdb.delete(ids=all_ids)
    except Exception:
        pass


@pytest.fixture
def human_llm_instance(vectordb_backend):
    """Create a HumanLLM instance with test configuration."""
    # Setup HumanLLMConfig
    config = HumanLLMConfig()
    config.common_vectordb = vectordb_backend
    
    # Setup LLM chains (using lightweight models for testing)
    llm_chains = {
        "default_llm": ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.0,
            max_tokens=4096,
            cache=False
        ),
        "premium_llm": ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", 
            temperature=0.0,
            max_tokens=4096,
            cache=False
        )
    }
    
    # Create HumanLLM instance
    human_llm = HumanLLM(
        agent_name="HamletFixerAgent",
        llmORchains_list=llm_chains,
        automation="full_auto",  # Disable manual interventions
        num_parallel_inferences=1,  # Will be overridden in tests
        temperature_min=0.,
        temperature_max=1.0
    )
    
    return human_llm


def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using difflib."""
    matcher = difflib.SequenceMatcher(None, text1.strip(), text2.strip())
    return matcher.ratio()


def measure_performance(func, *args, **kwargs) -> Tuple[any, float]:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


class TestGenerationSelectionTechniques:
    """Test suite for comparing generation and selection techniques."""
    # Class variable to store all test results
    test_results = []

    @pytest.mark.parametrize("generation_technique", [
        "temperature_variation",
        "self_refinement",  # Fixed syntax error
        "multi_experts",    
        "mixture_of_agents_generation",
        "iterative_alternatives"
    ])
    @pytest.mark.parametrize("selection_technique", [
        "concat",
        "best_of_n",
        "moa",
        "majority", 
        "last"
    ])
    @pytest.mark.parametrize("num_inferences", [
        5, 
        5, 
        5, 
        5, 
        5
        ])  # Reduced from [2, 3, 5] for faster testing
    @pytest.mark.parametrize("char_limit", [
        # 300, 
        2000
        ])
    def test_generation_selection_combinations(
        self,
        human_llm_instance,
        generation_technique,
        selection_technique,
        num_inferences,
        char_limit 
    ):
        """Test different combinations of generation and selection techniques."""
        
        # Configure the HumanLLM instance
        human_llm_instance.num_parallel_inferences = num_inferences
        human_llm_instance.selection_technique = selection_technique
        
        system_prompt = """Your task is to reorder the scrambled lines of this text excerpt from Shakespeare to restore their original sequence. When reordering, keep each line exactly as it appears. Return only the reordered text, with no explanations."""
        
        # Use shorter text for faster testing
        #short_hamlet = fetch_hamlet_text(char_limit)
        hamlet_extract_scrambled, reference_hamlet = get_scrambled_hamlet(char_limit)
        user_message = f"Please fix and improve this text from Hamlet:\n\n{hamlet_extract_scrambled}"
        
        # Create proper message objects
        original_input_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        # Measure performance
        try:
            result, execution_time = measure_performance(
                human_llm_instance.invoke,
                system_prompt_template=None,
                user_message=user_message,
                return_message_content_only=True,
                stream_output=False,
                generation_technique=generation_technique,
                selection_technique=selection_technique,
                original_input_messages=original_input_messages
            )
        except Exception as e:
            execution_time = 0.0
            result = ""
            print(f"Test failed with {generation_technique}/{selection_technique}: {e}")

        # Calculate similarity to reference
        if isinstance(result, list):
            fixed_text = result[0] if result else ""
        else:
            fixed_text = result or ""
            
        # Compare against the same-length reference snippet
        similarity_score = calculate_similarity_score(fixed_text, reference_hamlet)
        baseline_score = calculate_similarity_score(hamlet_extract_scrambled, reference_hamlet)

        preview = fixed_text if len(fixed_text) <= 200 else fixed_text[:200] + "..."
        # Store results for comparison
        test_result = {
            "generation_technique": generation_technique,
            "selection_technique": selection_technique,
            "num_inferences": num_inferences,
            "baseline_score": baseline_score,
            "similarity_score": similarity_score,
            "execution_time": execution_time,
            "output_length": len(fixed_text),
            "status": "success" if result else "failed",
            "fixed_text": preview,
            "char_limit": char_limit
        }
        
        self.__class__.test_results.append(test_result)

        # Basic assertions
        if test_result["status"] == "success":
            assert isinstance(result, (str, list)), "Result should be string or list"
            assert len(fixed_text) > 0, "Result should not be empty"
            assert similarity_score >= 0.0, "Similarity score should be non-negative"
            assert execution_time > 0, "Execution time should be positive"
            # print reference_hamlet vs fixed_text vs hamlet_extract_scrambled using a onliner
            print(f"Reference: {reference_hamlet[:200]}...")
            print(f"Fixed: {fixed_text[:200]}...")
            print(f"Scrambled: {hamlet_extract_scrambled[:200]}...")

        # Log results for analysis
        print(f"\n--- Test Results ---")
        print(f"Generation: {generation_technique}")
        print(f"Selection: {selection_technique}")
        print(f"Num Inferences: {num_inferences}")
        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Execution Time: {execution_time:.2f}s")
        print(f"Output Preview: {test_result['fixed_text']}")
        
        return test_result

    @classmethod
    def teardown_class(cls):
        """Called after all tests in the class are complete."""
        cls.display_results_table()
    
    @classmethod
    def display_results_table(cls):
        """Display comprehensive results table."""
        if not cls.test_results:
            print("No test results to display")
            return
            
        print("\n" + "="*100)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*100)
        
        # Prepare data for table display
        table_data = []
        for result in cls.test_results:
            table_data.append([
                result["generation_technique"],
                result["selection_technique"],
                result["num_inferences"],
                result["char_limit"],
                f"{result['baseline_score']:.4f}",       # new column
                f"{result['similarity_score']:.4f}",
                f"{result['execution_time']:.2f}s",
                result["output_length"],
                result["status"],
            ])

        headers = ["Generation", "Selection", "Inferences", "CharLimit", "Baseline", "Similarity", "Time", "Length", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Summary statistics
        successful_results = [r for r in cls.test_results if r["status"] == "success"]

        if successful_results:
            avg_similarity = sum(r["similarity_score"] for r in successful_results) / len(successful_results)
            avg_time = sum(r["execution_time"] for r in successful_results) / len(successful_results)
            
            print(f"\nSUMMARY:")
            print(f"Total tests: {len(cls.test_results)}")
            print(f"Successful: {len(successful_results)}")
            print(f"Average similarity: {avg_similarity:.4f}")
            print(f"Average execution time: {avg_time:.2f}s")
        else:
            print("No successful tests to summarize")

class TestFeedbackComparisonTechniques:
    """Test suite for comparing different feedback techniques with ensembling."""
    
    feedback_results = []
    
    @pytest.fixture
    def sample_inference_output(self):
        """Generate a sample output that needs improvement."""
        return """
def calculate_average(numbers):
    # Calculate average of numbers
    total = 0
    for num in numbers:
        total += num
    avg = total / len(numbers)
    return avg

# Example usage
result = calculate_average([1, 2, 3, 4, 5])
print(f"Average: {result}")
"""

    @pytest.fixture
    def expected_improvements(self):
        """Define expected improvements for validation."""
        return {
            "error_handling": ["try:", "except", "ZeroDivisionError"],
            "input_validation": ["if not numbers:", "if len(numbers) == 0:", "empty", "raise"],
            "type_hints": ["->", "List", "float", "numbers: List"],
            "docstring": ['"""', "Calculate", "Args:", "Returns:", "Raises:"],
            "better_names": ["sum_total", "count", "mean"],
            "edge_cases": ["nan", "inf", "None", "is None"]
        }

    def _validate_improvements(self, original: str, improved: str, expected_improvements: dict) -> dict:
        """Validate that improvements were actually made."""
        validation_results = {}
        
        for improvement_type, indicators in expected_improvements.items():
            original_has = any(indicator in original for indicator in indicators)
            improved_has = any(indicator in improved for indicator in indicators)
            validation_results[improvement_type] = {
                "was_present": original_has,
                "is_present": improved_has,
                "improved": improved_has and not original_has
            }
        
        return validation_results

    def _evaluate_code_improvement_detailed(self, original: str, improved: str, validation_results: dict) -> tuple:
        """Detailed evaluation of code improvement with validation."""
        score = 0.0
        improvements_made = []
        
        # Score based on actual improvements
        for improvement_type, result in validation_results.items():
            if result["improved"]:
                improvements_made.append(improvement_type)
                if improvement_type == "error_handling":
                    score += 0.25
                elif improvement_type == "input_validation":
                    score += 0.20
                elif improvement_type == "type_hints":
                    score += 0.15
                elif improvement_type == "docstring":
                    score += 0.20
                elif improvement_type == "better_names":
                    score += 0.10
                elif improvement_type == "edge_cases":
                    score += 0.10
        
        # Check if the code is still valid Python
        try:
            compile(improved, '<string>', 'exec')
            is_valid_python = True
        except:
            is_valid_python = False
            score *= 0.5  # Penalty for invalid Python
        
        return min(score, 1.0), improvements_made, is_valid_python

    @pytest.mark.parametrize("feedback_technique", [
        "annotations",           # Span-based annotations
        "instructions",          # General critiques
        "refinement",           # Iterative refinement
        "none"                  # No feedback (baseline)
    ])
    @pytest.mark.parametrize("generation_technique", [
        "temperature_variation",
        "multi_experts",
        "mixture_of_agents_generation"
    ])
    @pytest.mark.parametrize("selection_technique", [
        "best_of_n",
        "moa",
        "majority"
    ])
    @pytest.mark.parametrize("num_candidates", [1, 3, 5])
    def test_feedback_techniques_comparison_with_validation(
        self,
        human_llm_instance,
        sample_inference_output,
        expected_improvements,
        feedback_technique,
        generation_technique,
        selection_technique,
        num_candidates
    ):
        """Test different feedback techniques with deep validation."""
        
        start_time = time.time()
        
        try:
            if feedback_technique == "annotations":
                # Test span-based annotations
                feedback_result = human_llm_instance.generate_annotations_feedback(
                    inference_result_content=sample_inference_output,
                    output_id=1,
                    annotation_types="FIX, IMPROVE, DELETE",
                    annotation_number=5,
                    generation_technique=generation_technique,
                    selection_technique=selection_technique,
                    num_candidates=num_candidates
                )
                
                # Validate annotations format
                annotations = feedback_result['annotations']
                assert "\\FIX{" in annotations or "\\IMPROVE{" in annotations or "\\DELETE{" in annotations, \
                    "Annotations should contain proper format markers"
                
                # Apply the annotations
                improved_output = human_llm_instance.apply_feedback(
                    suggestions=feedback_result['annotations'],
                    text_content=sample_inference_output,
                    text_has_annotations=True,
                    annotation_format='latex-inline',
                    instruction_processing_approach='ANNOTATIONS_ALL'
                )
                
            elif feedback_technique == "instructions":
                # Test general critiques
                feedback_result = human_llm_instance.generate_instructions_feedback(
                    inference_result_content=sample_inference_output,
                    output_id=1,
                    generation_technique=generation_technique,
                    selection_technique=selection_technique,
                    num_candidates=num_candidates
                )
                
                # Validate suggestions format
                suggestions = feedback_result['suggestions']
                assert len(suggestions) > 10, "Suggestions should be meaningful"
                assert "improve" in suggestions.lower() or "fix" in suggestions.lower(), \
                    "Suggestions should contain improvement keywords"
                
                # Apply the suggestions
                improved_output = human_llm_instance.apply_feedback(
                    suggestions=feedback_result['suggestions'],
                    text_content=sample_inference_output,
                    text_has_annotations=False
                )
                
            elif feedback_technique == "refinement":
                # Test iterative refinement
                improved_output = sample_inference_output
                refinement_history = [sample_inference_output]
                
                for i in range(3):  # 3 refinement iterations
                    candidates = human_llm_instance.generate_candidates(
                        generation_technique="self_refinement",
                        system_prompt="Improve this Python code by making it more robust and efficient.",
                        user_prompt=improved_output,
                        num_responses=num_candidates,
                        use_premium_llm=True,
                        function_calling=False,
                        temp_min=0.0,
                        temp_max=0.5 if num_candidates > 1 else 0.0,
                        stream_output=False
                    )
                    if num_candidates > 1:
                        selected = human_llm_instance.select_candidate(candidates, selection_technique)
                        improved_output = selected[0].content if selected else improved_output
                    else:
                        improved_output = candidates[0].content if candidates else improved_output
                    
                    refinement_history.append(improved_output)
                
                # Validate progressive improvement
                for i in range(1, len(refinement_history)):
                    assert refinement_history[i] != refinement_history[i-1], \
                        f"Refinement iteration {i} should produce different output"
                        
            else:  # "none"
                # No feedback - baseline
                improved_output = sample_inference_output
                feedback_result = {"technique": "none"}
            
            execution_time = time.time() - start_time
            
            # Deep validation of improvements
            validation_results = self._validate_improvements(
                sample_inference_output,
                improved_output,
                expected_improvements
            )
            
            # Detailed evaluation
            improvement_score, improvements_made, is_valid_python = \
                self._evaluate_code_improvement_detailed(
                    sample_inference_output,
                    improved_output,
                    validation_results
                )
            
            # Store results
            result = {
                "feedback_technique": feedback_technique,
                "generation_technique": generation_technique,
                "selection_technique": selection_technique if num_candidates > 1 else "single",
                "num_candidates": num_candidates,
                "execution_time": execution_time,
                "improvement_score": improvement_score,
                "improvements_made": improvements_made,
                "validation_results": validation_results,
                "is_valid_python": is_valid_python,
                "original_length": len(sample_inference_output),
                "improved_length": len(improved_output),
                "status": "success",
                "is_ensemble": num_candidates > 1
            }
            
            # Additional validation assertions
            if feedback_technique != "none":
                assert improved_output != sample_inference_output, \
                    "Feedback should produce different output"
                assert is_valid_python, "Improved code should be valid Python"
                assert len(improvements_made) > 0, \
                    "At least one improvement should be made"
            
        except Exception as e:
            result = {
                "feedback_technique": feedback_technique,
                "generation_technique": generation_technique,
                "selection_technique": selection_technique if num_candidates > 1 else "single",
                "num_candidates": num_candidates,
                "execution_time": 0,
                "improvement_score": 0,
                "improvements_made": [],
                "validation_results": {},
                "is_valid_python": False,
                "original_length": len(sample_inference_output),
                "improved_length": 0,
                "status": f"failed: {str(e)}",
                "is_ensemble": num_candidates > 1
            }
            
        self.__class__.feedback_results.append(result)
        
        return result
    
    @pytest.mark.parametrize("test_case", [
        {
            "code": "def add(a, b): return a + b",
            "expected_improvements": ["type hints", "docstring", "error handling"]
        },
        {
            "code": "numbers = [1, 2, 3]\nfor i in range(len(numbers)): print(numbers[i])",
            "expected_improvements": ["pythonic iteration", "enumerate", "list comprehension"]
        },
        {
            "code": "def process_data(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result",
            "expected_improvements": ["list comprehension", "type hints", "docstring", "validation"]
        }
    ])
    def test_specific_improvement_cases(self, human_llm_instance, test_case):
        """Test specific code improvement scenarios."""
        code = test_case["code"]
        expected = test_case["expected_improvements"]
        
        # Generate feedback using best performing technique from previous tests
        feedback_result = human_llm_instance.generate_instructions_feedback(
            inference_result_content=code,
            generation_technique="multi_experts",
            selection_technique="moa",
            num_candidates=3
        )
        
        # Apply feedback
        improved_code = human_llm_instance.apply_feedback(
            suggestions=feedback_result['suggestions'],
            text_content=code,
            text_has_annotations=False,
            initial_prompt="Improve this Python code by making it more robust and efficient."
        )
        
        improved_stems = stem_phrase(improved_code)

        match_count = 0
        for improvement in expected:
            improvement_stems = stem_phrase(improvement)
            if any(word in improved_stems for word in improvement_stems):
                match_count += 1
        
        # Flexible threshold: at least 2 of the expected improvements
        min_matches_required = max(1, len(expected) - 1)
        assert match_count >= min_matches_required, (
            f"Only {match_count}/{len(expected)} expected improvements found.\n"
            f"Expected: {expected}\nImproved Code:\n{improved_code}"
        )
                
        # Ensure code is still executable
        try:
            # Clean improved_code to get only the code if there's a ```python block
            # if improved_code includes("```python"):
            if "```python" in improved_code:
                improved_code = improved_code.split("```python")[1].strip()
                improved_code = improved_code.split("```")[0].strip()
            compile(improved_code, '<string>', 'exec')
            is_valid = True
        except:
            is_valid = False
        
        assert is_valid, "Improved code should be valid Python"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])