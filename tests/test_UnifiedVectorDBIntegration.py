import json
import pytest
from unittest.mock import MagicMock, patch

# Import the classes under test.
# Adjust these import paths to match your project’s structure.
from utils.human_llm_config import HumanLLMConfig
from learn import PlannerAgent  # For add_learnt_task / add_failed_task
from learn import TaskIdentificationAgent
from utils.llm_utils import UnifiedVectorDB  # To reference the real class if needed

from config import *

# ------------------------------------------------------------------------------
# Fixture: patch out all UnifiedVectorDB constructors so they return the same MagicMock
# ------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_unifiedvectordb_constructor(monkeypatch):
    mock_uvdb = MagicMock(spec=UnifiedVectorDB)
    mock_uvdb.get_agent_data.return_value = ([], [])  # Fix for other tests

    # Any call to UnifiedVectorDB(...) returns our MagicMock
    monkeypatch.setattr("utils.human_llm_config.UnifiedVectorDB", lambda *args, **kwargs: mock_uvdb)
    monkeypatch.setattr("utils.db_utils.UnifiedVectorDB", lambda *args, **kwargs: mock_uvdb)
    return mock_uvdb

# ------------------------------------------------------------------------------
# 1. Test for HumanLLMConfig.log_agent_data
# ------------------------------------------------------------------------------
def test_log_agent_data_calls_add_texts_correctly(patch_unifiedvectordb_constructor):
    """
    - Ensure that log_agent_data serializes data_value → JSON string
    - Constructs metadata tags (including the provided metadata dict, plus agent_name & data_key)
    - Calls common_vectordb._add_texts(texts=[serialized_data], metadatas=[tags]).
    """
    hv = HumanLLMConfig()
    # Assign a dummy logger to avoid AttributeError
    hv.logger = MagicMock()

    # Ensure hv.common_vectordb is our patched MagicMock
    hv.common_vectordb = patch_unifiedvectordb_constructor

    mock_add_texts = MagicMock()
    hv.common_vectordb._add_texts = mock_add_texts

    # Call with a dict data_value and extra metadata
    hv.log_agent_data(
        agent_name="TestAgent",
        data_key="test_key",
        data_value={"foo": "bar"},
        function_name="test_fn",
        task_id=False,
        before_after="before",
        user_id="user123",
        step_id=42,
        task_type="task_type",
        score=0.75,
        metadata={"extra_meta": "extra_value"}
    )

    # Verify that the UnifiedVectorDB.log_agent_data was called, which should call _add_texts
    hv.common_vectordb.log_agent_data.assert_called_once_with(
        agent_name="TestAgent", data_key="test_key", data_value={"foo": "bar"},
        function_name="test_fn", task_id=False, before_after="before",
        user_id="user123", step_id=42, task_type="task_type", score=0.75,
        metadata={"extra_meta": "extra_value"}
    )

# ------------------------------------------------------------------------------
# 2. Test for HumanLLMConfig.get_agent_data (pagination & parsing)
# ------------------------------------------------------------------------------
def test_get_agent_data_pagination_and_parsing(patch_unifiedvectordb_constructor):
    """
    - get_agent_data builds metadata_filter from (agent_name, data_key, user_id)
    - Calls common_vectordb._query(query_text=..., metadata_filter=..., k=<passed k>, sort_order=...)
    - Paginates by start_index/end_index after retrieving results
    - JSON-parses page_content when new_storage=True
    """
    hv = HumanLLMConfig()
    hv.common_vectordb = patch_unifiedvectordb_constructor

    # Prepare two dummy "Document-like" return values
    fake_doc1 = MagicMock(page_content='{"k":"v1"}', metadata={"m": "x"})
    fake_doc2 = MagicMock(page_content='{"k":"v2"}', metadata={"m": "y"})

    # Make query return [fake_doc1, fake_doc2]
    patch_unifiedvectordb_constructor._query.return_value = [fake_doc1, fake_doc2]

    # Call get_agent_data with start_index=1; new_storage=True
    parsed, raw = hv.get_agent_data(
        agent_name="Agent1",
        data_key="k",
        user_id="user42",
        k=10,
        start_index=1,
        end_index=None,
        query_text="foo",
        new_storage=True
    )

    # It should have passed k=10 (unchanged) to query, not adjusted to 2
    patch_unifiedvectordb_constructor._query.assert_called_once_with(
        query_text="foo",
        metadata_filter={"agent_name": "Agent1", "data_key": "k", "user_id": "user42"},
        sort_order=None,
        k=10
    )

    # Since start_index=1, only fake_doc2 gets into `parsed`
    assert len(parsed) == 1
    assert parsed[0] == {"k": "v2"}    # JSON-parsed from fake_doc2.page_content
    assert raw == [fake_doc1, fake_doc2]

    # Now test new_storage=False branch: it should extract only the "k" field
    patch_unifiedvectordb_constructor._query.return_value = [fake_doc1]
    parsed2, raw2 = hv.get_agent_data(
        agent_name="AgentX",
        data_key="k",
        k=5,
        query_text="bar",
        new_storage=False
    )
    # Because new_storage=False, it returns ["v1"] (just the value of "k" from the JSON)
    assert parsed2 == ["v1"]
    assert raw2 == [fake_doc1]

# ------------------------------------------------------------------------------
# 3. Test for HumanLLMConfig.retrieve_logs
# ------------------------------------------------------------------------------
def test_retrieve_logs_calls_query_with_correct_filter(patch_unifiedvectordb_constructor):
    """
    retrieve_logs(agent_name, function_name, max_entries) must:
    - Call common_vectordb._query(query_text="*", metadata_filter={"function_name":..., "agent_name":...}, k=max_entries, sort_order="desc")
    - Return whatever query(...) returned
    """
    hv = HumanLLMConfig()
    hv.common_vectordb = patch_unifiedvectordb_constructor

    fake_return = ["docA", "docB"]
    patch_unifiedvectordb_constructor._query.return_value = fake_return

    result = hv.retrieve_logs(agent_name="A1", function_name="fn1", max_entries=5)
    patch_unifiedvectordb_constructor._query.assert_called_once_with(
        query_text="*",
        metadata_filter={"function_name": "fn1", "agent_name": "A1"},
        k=5,
        sort_order="desc"
    )
    assert result == fake_return

# ------------------------------------------------------------------------------
# 4. Tests for get_learnt_tasks and get_failed_tasks (parametrized)
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("method_name,history_getter", [
    ("get_learnt_tasks", "get_completed_tasks"),
    ("get_failed_tasks", "get_failed_tasks"),
])
def test_get_tasks_populates_task_history_and_returns_content(
    patch_unifiedvectordb_constructor, method_name, history_getter
):
    """
    For both get_learnt_tasks and get_failed_tasks:
    - Must set hv.common_vectordb_config.common_vectordb_embedding_function before calling
    - initialize_class_db() sets hv.db_learnt_tasks / hv.db_failed_tasks to our MagicMock
    - If similarity_search=False → call ._query(...)
    - If similarity_search=True  → call ._similarity_search_with_score(...)
    - Populate hv.task_history with each result.page_content
    - Return a set of page_content strings
    """
    hv = HumanLLMConfig()
    # Set a dummy embedding function so that initialize_class_db() won’t raise
    hv.common_vectordb_config.common_vectordb_embedding_function = lambda x: x
    hv.common_vectordb = patch_unifiedvectordb_constructor

    # Prepare two fake results
    fake1 = MagicMock(page_content="task1", metadata={})
    fake2 = MagicMock(page_content="task2", metadata={})

    # 4a. similarity_search=False branch
    patch_unifiedvectordb_constructor._query.return_value = [fake1, fake2]

    # Clear any existing history
    hv.task_history.clear_completed_tasks()
    hv.task_history.clear_failed_tasks()

    # Invoke get_learnt_tasks or get_failed_tasks with similarity_search=False
    ret_set = getattr(hv, method_name)(
        query_text="zzz",
        k=2,
        metadata_filter=None,
        sort_order="asc",
        similarity_search=False
    )
    # As similarity_search=False, ._query(...) must have been called
    patch_unifiedvectordb_constructor._query.assert_called_with(
        query_text="zzz",
        k=2,
        metadata_filter=None,
        sort_order="asc"
    )
    # The return value should be {"task1", "task2"}
    assert ret_set == {"task1", "task2"}

    # Confirm task_history filled appropriately
    history_list = getattr(hv.task_history, history_getter)()
    assert history_list == ["task1", "task2"]

    # 4b. similarity_search=True branch
    patch_unifiedvectordb_constructor._similarity_search_with_score.return_value = [fake1]
    ret_set2 = getattr(hv, method_name)(
        query_text="abc",
        k=1,
        metadata_filter=None,
        sort_order=None,
        similarity_search=True
    )
    patch_unifiedvectordb_constructor._similarity_search_with_score.assert_called_with(
        query="abc", k=1
    )
    # Return should be {"task1"}
    assert ret_set2 == {"task1"}

# ------------------------------------------------------------------------------
# 5. Test for HumanLLMConfig.get_validation_results
# ------------------------------------------------------------------------------
def test_get_validation_results_queries_common_vectordb_with_validationagent_filter(patch_unifiedvectordb_constructor):
    """
    get_validation_results must:
    - Set hv.common_vectordb_config.common_vectordb_embedding_function
    - Call initialize_class_db() internally
    - For similarity_search=False → call ._query(query_text, k, metadata_filter={"agent_name":"ValidationAgent"}, sort_order=...)
    - For similarity_search=True  → call ._similarity_search_with_score(query=..., k=..., metadata_filter={"agent_name":"ValidationAgent"})
    - Return a set of page_content strings
    """
    hv = HumanLLMConfig()
    # Pre-set embedding function
    hv.common_vectordb_config.common_vectordb_embedding_function = lambda x: x
    hv.common_vectordb = patch_unifiedvectordb_constructor

    fake = MagicMock(page_content="val123", metadata={})

    # similarity_search=False
    patch_unifiedvectordb_constructor._query.return_value = [fake]
    res = hv.get_validation_results(
        query_text="qqq",
        k=1,
        sort_order="desc",
        similarity_search=False
    )
    patch_unifiedvectordb_constructor._query.assert_called_once_with(
        query_text="qqq",
        k=1,
        metadata_filter={"agent_name": "ValidationAgent"},
        sort_order="desc"
    )
    assert res == {"val123"}

    # similarity_search=True
    patch_unifiedvectordb_constructor._similarity_search_with_score.return_value = [fake]
    res2 = hv.get_validation_results(
        query_text="xxx",
        k=2,
        similarity_search=True
    )
    patch_unifiedvectordb_constructor._similarity_search_with_score.assert_called_with(
        query="xxx",
        k=2,
        metadata_filter={"agent_name": "ValidationAgent"}
    )
    assert res2 == {"val123"}

# ------------------------------------------------------------------------------
# 6. Test for TaskIdentificationAgent.get_rag_documents
# ------------------------------------------------------------------------------
def test_get_rag_documents_applies_rag_filter(monkeypatch, patch_unifiedvectordb_constructor):
    """
    get_rag_documents (in TaskIdentificationAgent) should wrap config.get_agent_data(...)
    by forcing metadata_filter={"rag": True} (plus any extra_filter).
    """
    # Patch HumanLLMConfig.get_agent_data so we can track its call
    called = {}
    def fake_get_agent_data(agent_name, metadata_filter, query_text, **kwargs):
        called["agent_name"] = agent_name
        called["metadata_filter"] = metadata_filter
        called["query_text"] = query_text
        return (["dummy"], ["raw"])

    monkeypatch.setattr("utils.human_llm_config.HumanLLMConfig.get_agent_data", fake_get_agent_data)

    # Create a TaskIdentificationAgent (its config is a fresh HumanLLMConfig singleton)
    tia = TaskIdentificationAgent(
        default_llm_choice=None,
        envs=[],
        llmORchains_list={},
        problem_prompts_subdir=None
    )

    # Call get_rag_documents with an extra_filter that adds {"foo":"bar"}
    docs, raw = tia.get_rag_documents(
        agent_name="AgentX",
        extra_filter={"foo": "bar"},
        query="searchTerm"
    )

    # Since extra_filter={"foo":"bar"}, final metadata_filter should be {"rag": True, "foo": "bar"}
    assert called["metadata_filter"] == {"rag": True, "foo": "bar"}
    assert called["agent_name"] == "AgentX"
    assert called["query_text"] == "searchTerm"
    assert docs == ["dummy"]
    assert raw == ["raw"]

# ------------------------------------------------------------------------------
# 7. Tests for PlannerAgent.add_learnt_task / add_failed_task
# ------------------------------------------------------------------------------
def test_planneragent_add_tasks_calls_vectordb_add_texts(patch_unifiedvectordb_constructor):
    """
    - Instantiate PlannerAgent (its HumanLLMConfig will be used under the hood)
    - At construction time, nothing calls initialize_class_db, so db_learnt_tasks / db_failed_tasks remain None
    - Once add_learnt_task is called, it should create them (via initialize_class_db) and call add_texts(...)
    - Likewise for add_failed_task
    """
    # We must set the embedding function or PlannerAgent’s underlying config methods will balk
    HumanLLMConfig().common_vectordb_config.common_vectordb_embedding_function = lambda x: x

    pa = PlannerAgent(
        default_llm_choice=None,
        envs=[],
        premium_llm_choice=None,
        problem_prompts_subdir=None,
        skip_rounds=0,
        special_criteria=None,
        llmORchains_list={},
        model_choice=None,
        num_parallel_inferences=1,
        primitives_dir=None,
        system_prompt_path=None
    )

    # At this point, pa.db_learnt_tasks likely is None; calling add_learnt_task should trigger initialize_class_db
    pa.add_learnt_task(serialized_entry="entry1", tags={"tag1": "val1"})
    patch_unifiedvectordb_constructor._add_texts.assert_any_call(
        texts=["entry1"],
        metadatas=[{"tag1": "val1"}]
    )

    # Similarly for add_failed_task
    pa.add_failed_task(serialized_entry="entry2", tags={"tag2": "val2"})
    patch_unifiedvectordb_constructor._add_texts.assert_any_call(
        texts=["entry2"],
        metadatas=[{"tag2": "val2"}]
    )

# ------------------------------------------------------------------------------
# 8. Test that initialize_class_db creates two UnifiedVectorDB instances
# ------------------------------------------------------------------------------
def test_initialize_class_db_creates_two_vectordb_instances(patch_unifiedvectordb_constructor):
    """
    initialize_class_db() should set both db_learnt_tasks and db_failed_tasks to new UnifiedVectorDB(...) instances.
    Since we’ve globally patched UnifiedVectorDB to always return the same MagicMock,
    both attributes should reference patch_unifiedvectordb_constructor.
    """
    hv = HumanLLMConfig()
    # Pre-set embedding function
    hv.common_vectordb_config.common_vectordb_embedding_function = lambda x: x

    # Before calling, these should be None
    assert hv.db_learnt_tasks is None
    assert hv.db_failed_tasks is None

    # Now invoke initialize_class_db
    hv.initialize_class_db()

    # Both should now point to our shared MagicMock
    assert hv.db_learnt_tasks is patch_unifiedvectordb_constructor
    assert hv.db_failed_tasks is patch_unifiedvectordb_constructor

# ------------------------------------------------------------------------------
# 9. Test TaskIdentificationAgent.identify_best_task triggers get_learnt_tasks & get_failed_tasks
# ------------------------------------------------------------------------------
def test_identify_best_task_builds_prompt_and_calls_get_tasks(monkeypatch, patch_unifiedvectordb_constructor):
    """
    We only verify that identify_best_task() calls get_learnt_tasks() and get_failed_tasks() on HumanLLMConfig.
    We monkeypatch these methods to avoid invoking actual LLM logic.
    """
    # Monkeypatch get_learnt_tasks / get_failed_tasks on HumanLLMConfig
    called = {"learnt": False, "failed": False}

    def fake_get_learnt_tasks(*args, **kwargs):
        called["learnt"] = True
        return {"t1"}, []

    def fake_get_failed_tasks(*args, **kwargs):
        called["failed"] = True
        return {"f1"}, []

    monkeypatch.setattr("utils.human_llm_config.HumanLLMConfig.get_learnt_tasks", fake_get_learnt_tasks)
    monkeypatch.setattr("utils.human_llm_config.HumanLLMConfig.get_failed_tasks", fake_get_failed_tasks)

    # Create an environment whose get_state() returns a string
    dummy_env = MagicMock(get_state=lambda: "env_state")

    tia = TaskIdentificationAgent(
        default_llm_choice=None,
        envs=[dummy_env],
        llmORchains_list={},
        problem_prompts_subdir=None
    )
    # Assign dummy logger on the TaskIdentificationAgent so identify_best_task doesn’t error on logging
    tia.logger = MagicMock()

    # Call identify_best_task; we don't need its return value, only that it invoked both getters
    _ = tia.identify_best_task()

    assert called["learnt"], "get_learnt_tasks() was not called"
    assert called["failed"], "get_failed_tasks() was not called"

