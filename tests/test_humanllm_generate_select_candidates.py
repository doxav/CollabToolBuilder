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
            cache=False
        ),
        "premium_llm": ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", 
            temperature=0.0,
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
        #"concat",
        "best_of_n",
        "moa",
        "majority", 
        "last"
    ])
    @pytest.mark.parametrize("num_inferences", [
        2, 
        3, 
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

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])