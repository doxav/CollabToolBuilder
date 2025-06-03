# tests/test_human_llmconfig_db_integration.py

import json
import time
import pytest

from utils.human_llm_config import HumanLLMConfig
from utils.llm_utils import CHROMA_DATABASE, ELASTIC_DATABASE
from utils.llm_utils import UnifiedVectorDB, UnifiedVectorDBConfig

# Re‐use the `backend` fixture defined in tests/test_UnifiedVectorDB.py
from tests.test_UnifiedVectorDB import backend as vectordb_backend

# tests/test_UnifiedVectorDB.py
import pytest
from utils.llm_utils import (
    UnifiedVectorDB,
    UnifiedVectorDBConfig,
    CHROMA_DATABASE,
    ELASTIC_DATABASE,
)
from config import *
import time

class DummyEmbeddingNonZero:
    def __init__(self, dim: int = 16): self.dim = dim
    def embed_query(self, text: str): return [1.0]*self.dim
    def embed_documents(self, texts): return [[1.0]*self.dim for _ in texts]

def is_elasticsearch_available(es_url: str):
    try:
        from elasticsearch import Elasticsearch
        client = Elasticsearch([es_url], verify_certs=False, timeout=5)
        return client.ping()
    except Exception:
        return False

@pytest.fixture(params=[CHROMA_DATABASE, ELASTIC_DATABASE])
def backend(request, tmp_path):
    """
    Spins up a real UnifiedVectorDB for each of Chroma/Elasticsearch. 
    Passing `reset_indices=True` ensures a clean collection.
    """
    db_type = request.param
    config = UnifiedVectorDBConfig(
        embedding_function=DummyEmbeddingNonZero(dim=16),
        collection_name=f"test_collection_{db_type}",
        persist_directory=str(tmp_path / f"vectordb_data_{db_type}"),
        reset_indices=True
    )
    config.db_type = db_type

    # This actually instantiates a Chroma or ES store under the hood
    uvdb = UnifiedVectorDB(config=config, check_db=True)
    yield uvdb

    # Teardown: clear everything after each test
    try:
        if uvdb.config.db_type == CHROMA_DATABASE:
            all_ids = [doc.id for doc in uvdb.db._collection.get()]
            if all_ids:
                uvdb.delete(ids=all_ids)
        else:  # ELASTIC_DATABASE
            try:
                uvdb.clear()
            except Exception:
                pass
    except Exception:
        pass

@pytest.fixture(autouse=True)
def reset_singleton():
    """
    HumanLLMConfig is a singleton—clear it before each test run so that
    a new config (and new vector DB) is isolated.
    """
    # Force the singleton to re‐instantiate
    HumanLLMConfig._HumanLLMConfig__instance = None
    yield
    # (Cleanup happens automatically in vectordb_backend teardown)


def test_log_agent_data_persists_document_and_metadata(backend: UnifiedVectorDB):
    """
    1. Create a fresh HumanLLMConfig, set its common_vectordb to `backend`.
    2. Call log_agent_data(...) with a small dict and some metadata.
    3. Sleep briefly (ES might need a moment to index).
    4. Query the same `backend` for all docs where metadata.agent_name == "TestAgent"
       and metadata.data_key == "test_key", and ensure we retrieve exactly that JSON.
    """
    hv = HumanLLMConfig()
    hv.common_vectordb = backend

    # Use a small dictionary as data_value
    hv.log_agent_data(
        agent_name="TestAgent",
        data_key="test_key",
        data_value={"foo": "bar"},
        function_name="unit_test_fn",
        id_task=False,
        before_after="after",
        user_id="user42",
        step_id=7,
        type_tache="unit_test_type",
        score=0.123,
        metadata={"extra_meta": "xyz"}
    )

    # Give Elasticsearch a moment to index (Chroma is in‐memory and immediate)
    time.sleep(0.5)

    # Now query the vector DB for everything matching our metadata:
    results = backend.query(
        query_text="*",
        metadata_filter={
            "agent_name": "TestAgent",
            "data_key": "test_key"
        },
        k=10,
        sort_order=None
    )
    # We expect exactly one document back
    assert len(results) == 1

    doc = results[0]
    # Its page_content should be the JSON we serialized:
    retrieved_dict = json.loads(doc.page_content)
    assert retrieved_dict == {"foo": "bar"}

    # Check that all metadata fields were stored
    stored_meta = doc.metadata
    assert stored_meta["agent_name"] == "TestAgent"
    assert stored_meta["data_key"] == "test_key"
    assert stored_meta["function_name"] == "unit_test_fn"
    assert stored_meta["before_after"] == "after"
    assert stored_meta["user_id"] == "user42"
    assert stored_meta["step_id"] == 7
    assert stored_meta["type_tache"] == "unit_test_type"
    assert abs(stored_meta["score"] - 0.123) < 1e-6
    assert stored_meta["extra_meta"] == "xyz"
    assert "date" in stored_meta

@pytest.fixture(autouse=True)
def reset_human_llmconfig_singleton():
    # Always reset the singleton before each test so we start fresh
    HumanLLMConfig._HumanLLMConfig__instance = None
    yield


def test_get_agent_data_returns_json_documents(backend: UnifiedVectorDB):
    """
    1. Use backend.add_texts(...) to insert 3 JSON docs (under "agentX"/"k" with user_id="u1").  
    2. Call get_agent_data(agent_name="agentX", data_key="k", user_id="u1", k=10, start_index=1, new_storage=True).  
    3. We expect only the 2nd & 3rd documents (due to start_index=1).  
    4. Confirm that the returned list of dicts matches those JSON payloads.  
    """
    hv = HumanLLMConfig()
    hv.common_vectordb = backend

    # Prepare three JSON‐serializable payloads
    docs_to_insert = [
        {"k": "value_1", "other": "A"},
        {"k": "value_2", "other": "B"},
        {"k": "value_3", "other": "C"},
    ]
    # Corresponding metadatas must include agent_name, data_key, user_id
    metadatas = [
        {"agent_name": "agentX", "data_key": "k", "user_id": "u1"},
        {"agent_name": "agentX", "data_key": "k", "user_id": "u1"},
        {"agent_name": "agentX", "data_key": "k", "user_id": "u1"},
    ]
    texts = [json.dumps(d) for d in docs_to_insert]
    ids = [f"doc_{i}" for i in range(3)]

    # Bulk‐add into the real vector DB
    backend.add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(0.5)

    # Now call get_agent_data with start_index=1, new_storage=True
    parsed_list, raw_list = hv.get_agent_data(
        agent_name="agentX",
        data_key="k",
        user_id="u1",
        k=10,
        start_index=1,      # skip the first
        end_index=None,
        query_text="*",
        new_storage=True
    )
    # We expect exactly 2 documents returned (value_2, value_3), parsed as dicts:
    assert len(parsed_list) == 2
    expected_dicts = [{"k": "value_2", "other": "B"}, {"k": "value_3", "other": "C"}]
    assert parsed_list == expected_dicts

    # raw_list should be a list of the original Document‐like objects
    assert len(raw_list) == 3
    # Each `.page_content` in raw_list is still the JSON string; check the first two:
    assert json.loads(raw_list[0].page_content) == {"k": "value_1", "other": "A"}
    assert json.loads(raw_list[1].page_content) == {"k": "value_2", "other": "B"}


def test_get_agent_data_extracts_single_value_when_new_storage_false(backend: UnifiedVectorDB):
    """
    - Insert two JSON docs with data_key="k".  
    - Call get_agent_data(new_storage=False).  
    - Expect to receive a list of just the `"k"` values from each document.
    """
    hv = HumanLLMConfig()
    hv.common_vectordb = backend

    # Insert two JSON docs
    docs_to_insert = [
        {"k": "X1", "z": "foo"},
        {"k": "X2", "z": "bar"},
    ]
    metadatas = [
        {"agent_name": "a1", "data_key": "k"},
        {"agent_name": "a1", "data_key": "k"},
    ]
    texts = [json.dumps(d) for d in docs_to_insert]
    ids = ["one", "two"]

    backend.add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name="a1",
        data_key="k",
        k=10,
        query_text="*",
        new_storage=False
    )
    # Since new_storage=False, parsed should be ["X1", "X2"]
    assert parsed == ["X1", "X2"]
    assert len(raw) == 2
    assert json.loads(raw[0].page_content)["z"] == "foo"
    assert json.loads(raw[1].page_content)["z"] == "bar"

# tests/test_human_llmconfig_db_integration_extended.py

import json
import time
import pytest

from utils.human_llm_config import HumanLLMConfig
from utils.llm_utils import UnifiedVectorDB
# Import the shared backend fixture (Chroma & Elasticsearch)
from tests.test_UnifiedVectorDB import backend as vectordb_backend

@pytest.fixture(autouse=True)
def reset_human_llmconfig_singleton():
    """
    Always reset the HumanLLMConfig singleton so each test gets a fresh instance
    (with no prior common_vectordb, step_id, etc.).
    """
    HumanLLMConfig._HumanLLMConfig__instance = None
    yield

# ----------------------------------------------------------------------------------
# Table of all data_keys, example data, functions using them, and associated pytest:
#
# | Data Key                     | Example Payload                                                      | Function(s) Using It                                     | Pytest Name                                           |
# |------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------|-------------------------------------------------------|
# | saved_task                   | {"agent_name":"A","before_after":"before","content": {...}}           | prepare_configs / run_planner (via special_criteria)      | test_saved_task_log_and_retrieve                      |
# | rag_knowledge                | ["doc1","doc2"]                                                       | get_rag_documents (TaskIdentificationAgent)               | test_rag_knowledge_log_and_retrieve                   |
# | llm_suggestions              | ["suggest1","suggest2"]                                                | (Presumably logged by some agent for suggestions)         | test_llm_suggestions_log_and_retrieve                 |
# | llm_annotations              | [{"annot":"…"}]                                                       | (Presumably logged for LLM annotations)                   | test_llm_annotations_log_and_retrieve                 |
# | error_patches                | [("err_msg","diff_text"), …]                                           | code_task_and_run_test (during coding)                    | test_error_patches_log_and_retrieve                    |
# | unique_codes                 | ["code1","code2"]                                                      | coding loop dedup logic                                    | test_unique_codes_log_and_retrieve                     |
# | previous_scores              | [[{"scoreA":1}], [{"scoreB":2}]]                                       | coding loop (on each iteration)                           | test_previous_scores_log_and_retrieve                  |
# | previous_codes               | ["print('A')","print('B')"]                                            | coding loop                                                | test_previous_codes_log_and_retrieve                   |
# | previous_errors              | [["err1"], ["err2"]]                                                   | coding loop                                                | test_previous_errors_log_and_retrieve                  |
# | successful_codes             | [( {"program_code":"…"}, "feedback", {"sc":0.9} ), …]                   | when validation succeeds                                   | test_successful_codes_log_and_retrieve                 |
# | all_results                  | [ (parsed_code, bool, exec_res, {"sc":0.5}, [env_state], runtime), …]   | code loop accumulates all results                          | test_all_results_log_and_retrieve                      |
# | "percentage_no_runtime_error" | 0.75 (float)                                                          | returned by code_task_and_run_test → all_scores["…"]       | **Not stored in DB via log_agent_data** (no test)      |
# | "best_score_without_validation" | 0.85 (float)                                                       | returned by code_task_and_run_test → all_scores["…"]       | **Not stored in DB via log_agent_data** (no test)      |
# | "validated_scores"           | {"sc1":0.5,"sc2":0.7}                                                  | returned by code_task_and_run_test → all_scores["…"]       | **Not stored in DB via log_agent_data** (no test)      |
#
# **Notes on metrics keys:**  
# - Although the code *calculates* "percentage_no_runtime_error", "best_score_without_validation", and "validated_scores" inside 
#   `code_task_and_run_test`, these are never passed into `log_agent_data`. Instead, they are *returned* as part of `all_scores`. 
#   They do not appear in the vector DB, so we do **not** write a `log_agent_data`/`get_agent_data` test for them.
#
# For each of the above “logged” keys, the test below will:
#  1. Instantiate fresh HumanLLMConfig()  
#  2. Set `.common_vectordb = backend`  
#  3. Call `hv.log_agent_data(agent_name, data_key, example_payload, metadata={"test_meta":"X"})`  
#  4. `time.sleep(0.5)` (ES indexing)  
#  5. Call `hv.get_agent_data(agent_name, data_key, metadata_filter={"test_meta":"X"})`  
#  6. Assert that the parsed output matches exactly the example payload.
# ----------------------------------------------------------------------------------

# 1. saved_task
def test_saved_task_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "MyAgent"
    data_key = "saved_task"
    example_payload = {
        "agent_name": agent,
        "before_after": "before",
        "content": {"foo": "bar", "num_parallel_inferences": 3}
    }
    # Log under metadata {"test_meta":"X"}
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      function_name="fn", id_task=False, before_after="before",
                      user_id="u1", step_id=1, type_tache="t1", score=0.5,
                      metadata={"test_meta": "X"})

    time.sleep(0.5)

    # Retrieve it back
    parsed_list, raw_list = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"test_meta": "X"},
        k=5,
        query_text="*",
        new_storage=True
    )

    # We expect exactly one payload, parsed as a dict
    assert len(parsed_list) == 1
    assert parsed_list[0] == example_payload

# 2. rag_knowledge
def test_rag_knowledge_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentRAG"
    data_key = "rag_knowledge"
    example_payload = ["doc1", "doc2", "doc3"]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"batch": "alpha"})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"batch": "alpha"},
        k=3,
        query_text="*",
        new_storage=True
    )

    # Since non-dict data_values are wrapped as {data_key: data_value}, we expect:
    assert parsed == [{ data_key: example_payload }]
    assert parsed[0][data_key] == example_payload

# 3. llm_suggestions
def test_llm_suggestions_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentLLM"
    data_key = "llm_suggestions"
    example_payload = ["suggestionA", "suggestionB"]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"case": "test_sugg"})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"case": "test_sugg"},
        k=2,
        query_text="*",
        new_storage=True
    )

    assert parsed == [{ data_key: example_payload }]
    assert parsed[0][data_key] == example_payload

# 4. llm_annotations
def test_llm_annotations_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentAnnot"
    data_key = "llm_annotations"
    example_payload = [{"line": 1, "comment": "Fix this"}, {"line": 2, "comment": "Check that"}]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"case": "annot_case"})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"case": "annot_case"},
        k=2,
        query_text="*",
        new_storage=True
    )

    assert parsed == [{ data_key: example_payload }]
    assert parsed[0][data_key] == example_payload

# 5. error_patches
def test_error_patches_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentCode"
    data_key = "error_patches"
    example_payload = [("RuntimeError: x", "diff: fix x→y"), ("ValueError: z", "diff: fix z→w")]
    # `log_agent_data` will JSON‐serialize the list of tuples as a JSON‐string impersonating Python list of lists.
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"round": 1})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"round": 1},
        k=2,
        query_text="*",
        new_storage=True
    )

    # JSON‐parsed version of [("RuntimeError: x","diff:…"), …] → becomes a list of lists, not tuples
    assert parsed == [{ data_key: [["RuntimeError: x", "diff: fix x→y"], ["ValueError: z", "diff: fix z→w"]] }]
    assert parsed[0][data_key] == [["RuntimeError: x","diff: fix x→y"], ["ValueError: z","diff: fix z→w"]]

# 6. unique_codes
def test_unique_codes_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentCodeUnique"
    data_key = "unique_codes"
    example_payload = ["print('A')", "print('B')"]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"round": 2})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"round": 2},
        k=2,
        query_text="*",
        new_storage=True
    )

    assert parsed == [{ data_key: example_payload }]
    assert parsed[0][data_key] == example_payload

# 7. previous_scores
def test_previous_scores_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentPrevScores"
    data_key = "previous_scores"
    # Example: a list of score‐dicts per attempt
    example_payload = [{"accuracy": 0.8, "coverage": 0.6}, {"accuracy": 0.9, "coverage": 0.7}]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"prev_round": 3})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"prev_round": 3},
        k=2,
        query_text="*",
        new_storage=True
    )

    assert parsed == [{ data_key: example_payload }]
    assert parsed[0][data_key] == example_payload

# 8. previous_codes
def test_previous_codes_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentPrevCodes"
    data_key = "previous_codes"
    example_payload = ["print('X')", "print('Y')"]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"prev_round": 4})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"prev_round": 4},
        k=2,
        query_text="*",
        new_storage=True
    )

    assert parsed == [{ data_key: example_payload }]
    assert parsed[0][data_key] == example_payload

# 9. previous_errors
def test_previous_errors_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentPrevErrs"
    data_key = "previous_errors"
    # Example: a list of lists of error‐messages or AIMessage‐like objects
    example_payload = [["SyntaxError: a"], ["TypeError: b", "ValueError: c"]]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"prev_round": 5})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"prev_round": 5},
        k=2,
        query_text="*",
        new_storage=True
    )

    assert parsed == [{"previous_errors": example_payload}]
    assert parsed[0]["previous_errors"] == example_payload

# 10. successful_codes
def test_successful_codes_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentSuccess"
    data_key = "successful_codes"
    # Example: list of tuples (parsed_code_dict, feedback_str, score_dict)
    example_payload = [
        ({"program_code": "print('ok')"}, "Good", {"score": 0.95}),
        ({"program_code": "print('also_ok')"}, "Great", {"score": 0.98})
    ]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"stage": "validate"})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"stage": "validate"},
        k=2,
        query_text="*",
        new_storage=True
    )

    # JSON‐parsing a list of tuples → yields a list of lists of dict/string/dict
    expected = [
        [
            {"program_code": "print('ok')"}, "Good", {"score": 0.95}
        ],
        [
            {"program_code": "print('also_ok')"}, "Great", {"score": 0.98}
        ]
    ]

    assert parsed == [{ data_key: expected }]
    assert parsed[0][data_key] == expected

# 11. all_results
def test_all_results_log_and_retrieve(vectordb_backend: UnifiedVectorDB):
    hv = HumanLLMConfig()
    hv.common_vectordb = vectordb_backend

    agent = "AgentAllResults"
    data_key = "all_results"
    # Example: list of 2 “result‐tuples” (mimicking code_task_and_run_test output)
    example_payload = [
        ({"program_code": "c1"}, True, "exec1", {"scoreA": 0.4}, ["state1"], 0.12),
        ({"program_code": "c2"}, False, "exec2", {"scoreB": 0.6}, ["state2"], 0.25)
    ]
    hv.log_agent_data(agent_name=agent, data_key=data_key, data_value=example_payload,
                      metadata={"batch": "all"})

    time.sleep(0.5)

    parsed, raw = hv.get_agent_data(
        agent_name=agent,
        data_key=data_key,
        metadata_filter={"batch": "all"},
        k=2,
        query_text="*",
        new_storage=True
    )

    # After JSON parse, the two‐tuple‐of‐tuples becomes lists of lists of mixed types
    expected = [
        [
            {"program_code": "c1"}, True, "exec1", {"scoreA": 0.4}, ["state1"], 0.12
        ],
        [
            {"program_code": "c2"}, False, "exec2", {"scoreB": 0.6}, ["state2"], 0.25
        ]
    ]

    assert parsed == [{ data_key: expected }]
    assert parsed[0][data_key] == expected

# ----------------------------------------------------------------------------------
# Note on “percentage_no_runtime_error”, “best_score_without_validation”, “validated_scores”:
#
# These three metrics are computed inside `code_task_and_run_test` and returned in a Python dict (`all_scores`), but
# they are *not* sent to `log_agent_data`. Therefore, there is no database entry under those keys, and no get_agent_data
# call will ever retrieve them from the vector‐store. As a result, we do NOT write `log_agent_data`/`get_agent_data` tests 
# for them, since they never go into the vector database. They remain purely in‐memory return‐values in your code.
# ----------------------------------------------------------------------------------
