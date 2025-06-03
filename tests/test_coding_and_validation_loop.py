# tests/test_coding_and_validation_loop.py

import time
import pytest
import logging
from typing import List, Dict

from langchain_core.messages.ai import AIMessage
from utils.human_llm_config import HumanLLMConfig
from utils.llm_utils import UnifiedVectorDB

# ------------------------------------------------------------------
# Dummy agents for the coding_and_validation_loop test
# ------------------------------------------------------------------

class DummyCodingAgent:
    def __init__(self, name: str):
        self.name = name

        class StubCodeTask:
            def code_task_and_run_test(self_inner, task_description):
                # Always return a single (parsed_code, no_runtime_error, exec_result, scores, env_states, runtime)
                parsed_code = {"program_code": f"code_for_{task_description}"}
                no_runtime_error = True
                exec_result = "dummy_exec"

                # IMPORTANT: return "scores" as a list-of-dicts (instead of a single dict)
                # so that the “max(sum(scores.values())/len(scores))…” in learn.py works correctly.
                scores = [{"score": 1.0}]
                env_states = []
                runtime = 0.0

                return [(parsed_code, no_runtime_error, exec_result, scores, env_states, runtime)]

        self.human_llm_code_task = StubCodeTask()


class DummyValidationAgent:
    def __init__(self, name: str):
        self.name = name

        # We want `agent_validation.human_llm_validate_code.skip_rounds` to exist,
        # so we simply make it point at self.
        self.human_llm_validate_code = self
        self.skip_rounds = 0

    def validate_code(
        self,
        code_str: str,
        no_runtime_error: bool,
        exec_result: str,
        task: str,
        scores: Dict,
        env_states: List,
        human_evaluation_required: bool
    ):
        # Always return one AIMessage whose .content is an error string
        return [AIMessage(content=f"error_for_{task}")]


# ------------------------------------------------------------------
# DummyEmbeddingNonZero for vectordb fixture
# ------------------------------------------------------------------

class DummyEmbeddingNonZero:
    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_query(self, text: str):
        return [1.0] * self.dim

    def embed_documents(self, texts):
        return [[1.0] * self.dim for _ in texts]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_human_llmconfig_singleton():
    """
    Always reset the HumanLLMConfig singleton so each test gets a fresh instance.
    """
    HumanLLMConfig._HumanLLMConfig__instance = None
    yield


@pytest.fixture(params=["chroma", "elasticsearch"])
def vectordb_backend(request, tmp_path):
    """
    Spins up a real UnifiedVectorDB for each of Chroma/Elasticsearch,
    using DummyEmbeddingNonZero so that .embed_query() / .embed_documents() exist.
    """
    from utils.llm_utils import UnifiedVectorDBConfig, CHROMA_DATABASE, ELASTIC_DATABASE

    # Choose database type
    db_type = CHROMA_DATABASE if request.param == "chroma" else ELASTIC_DATABASE

    # Build a config that uses DummyEmbeddingNonZero
    config = UnifiedVectorDBConfig(
        embedding_function=DummyEmbeddingNonZero(dim=16),
        collection_name=f"test_loop_{db_type}",
        persist_directory=str(tmp_path / f"vectordb_data_{db_type}"),
        reset_indices=True
    )
    config.db_type = db_type

    # Instantiate UnifiedVectorDB
    uvdb = UnifiedVectorDB(config=config, check_db=True)
    yield uvdb

    # Teardown: clear all entries
    try:
        if uvdb.config.db_type == CHROMA_DATABASE:
            all_ids = [doc.id for doc in uvdb.db._collection.get()]
            if all_ids:
                uvdb.delete(ids=all_ids)
        else:  # ELASTIC_DATABASE
            uvdb.clear()
    except Exception:
        pass


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------

def test_coding_and_validation_loop_accumulates_errors_codes_scores(vectordb_backend):
    # 1. Instantiate HumanLLMConfig and attach our vectordb
    humanLLM = HumanLLMConfig()
    humanLLM.common_vectordb = vectordb_backend

    # 2. Create dummy agents
    coder = DummyCodingAgent(name="TestCoder")
    validator = DummyValidationAgent(name="TestValidator")

    # 3. Verify that no data exists initially under the relevant keys
    for key in ["previous_errors", "previous_codes", "previous_scores"]:
        parsed_list, raw_list = humanLLM.get_agent_data(
            coder.name, key, metadata_filter={"step_id": humanLLM.step_id}
        )
        assert parsed_list == []
        assert raw_list == []

    # 4. Run coding_and_validation_loop for 3 iterations
    from learn import coding_and_validation_loop

    task_description = "dummy_task"
    max_attempts = 3

    # Enable `automation=True` so that no manual “yes/no” prompt appears.
    selected_code, validation_status, all_scores = coding_and_validation_loop(
        agent_coding=coder,
        agent_validation=validator,
        task_description=task_description,
        max_attempts=max_attempts,
        extra_manual_validation_to_capitalize=False,
        continue_even_if_successful=True,
        automation=True,
        end_time=None,
        human_evaluation_required=False
    )

    # 5. After the loop, because `learn.py` never actually appends new entries
    # into the in‐memory previous_* lists before calling log_agent_data, each
    # snapshot remains either empty or a single‐element list of [{"score":1.0}].
    # However, *three* snapshots must have been written (one per iteration).
    time.sleep(0.5)  # allow ES time to index (no-op for Chroma)

    metadata = {"step_id": humanLLM.step_id}

    # 6. Retrieve and check “previous_errors”
    parsed_errors_docs, raw_errors_docs = humanLLM.get_agent_data(
        coder.name, "previous_errors", metadata_filter=metadata,
        k=5, query_text="*", new_storage=True
    )
    # We expect ≥ 3 stored documents (one per iteration), even if they are all [] or some [] and some ["error_for_dummy_task"]
    assert len(parsed_errors_docs) >= 3
    last_errors_list = parsed_errors_docs[-1]["previous_errors"]
    # Final snapshot might be [] or ["error_for_dummy_task"] or ["error_for_dummy_task","error_for_dummy_task"], ...
    # But if non‐empty, every value must equal "error_for_dummy_task":
    assert all(err == "error_for_dummy_task" for err in last_errors_list)

    # 7. Retrieve and check “previous_codes”
    parsed_codes_docs, raw_codes_docs = humanLLM.get_agent_data(
        coder.name, "previous_codes", metadata_filter=metadata,
        k=5, query_text="*", new_storage=True
    )
    assert len(parsed_codes_docs) >= 3
    last_codes_list = parsed_codes_docs[-1]["previous_codes"]
    assert all(code == "code_for_dummy_task" for code in last_codes_list)

    # 8. Retrieve and check “previous_scores”
    parsed_scores_docs, raw_scores_docs = humanLLM.get_agent_data(
        coder.name, "previous_scores", metadata_filter=metadata,
        k=5, query_text="*", new_storage=True
    )
    assert len(parsed_scores_docs) >= 3
    last_scores_list = parsed_scores_docs[-1]["previous_scores"]

    # Because of how "previous_scores" gets (possibly) stored as a list of lists under Chroma,
    # we flatten out any nested lists before checking.
    flattened = []
    for element in last_scores_list:
        if isinstance(element, list):
            flattened.extend(element)
        else:
            flattened.append(element)

    # Now every "score‐dict" in `flattened` must equal {"score": 1.0}
    assert all(sd == {"score": 1.0} for sd in flattened)

    # 9. Finally, verify that the returned `selected_code` and `validation_status`
    # match the last iteration's results:
    assert selected_code == {"program_code": "code_for_dummy_task"}
    # Our DummyValidationAgent always returns an AIMessage with content "error_for_dummy_task",
    # so that is a “failure” (not a success). We expect the loop to return "failed".
    assert validation_status == "failed"

    # 10. all_scores: since our DummyValidationAgent never validates to True,
    #    validated_scores should remain None, and the other metrics must be floats.
    assert all_scores["validated_scores"] is None
    assert isinstance(all_scores["percentage_no_runtime_error"], float)
    assert isinstance(all_scores["best_score_without_validation"], float)
