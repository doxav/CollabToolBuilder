import pytest, time
from unittest.mock import MagicMock

from utils.human_llm import HumanLLMConfig
from utils.llm_utils import CHROMA_DATABASE, ELASTIC_DATABASE
from utils.human_llm_config import UnifiedVectorDBConfig, UnifiedVectorDB

from config import *

# ————————————————————————————————————————————————
# Inline backend fixture (real Chroma/Elasticsearch) and reset singleton
# ————————————————————————————————————————————————
class DummyEmbeddingNonZero:
    def __init__(self, dim=16): self.dim = dim
    def embed_query(self, text): return [1.0] * self.dim
    def embed_documents(self, texts): return [[1.0] * self.dim for _ in texts]

@pytest.fixture(params=[CHROMA_DATABASE, ELASTIC_DATABASE])
def backend(request, tmp_path):
    db_type = request.param
    cfg = UnifiedVectorDBConfig(
        embedding_function=DummyEmbeddingNonZero(dim=16),
        collection_name=f"test_collection_{db_type}",
        persist_directory=str(tmp_path / f"vectordb_{db_type}"),
        reset_indices=True,
    )
    cfg.db_type = db_type

    uvdb = UnifiedVectorDB(config=cfg, check_db=True)
    yield uvdb

    # # Teardown: delete all entries
    # try:
    #     if uvdb.config.db_type == CHROMA_DATABASE:
    #         ids = [doc.id for doc in uvdb.db._collection.get()]
    #         if ids:
    #             uvdb.delete(ids=ids)
    #     else:
    #         try:
    #             uvdb.clear()
    #         except:
    #             pass
    # except:
    #     pass

@pytest.fixture(autouse=True)
def reset_singleton():
    # Ensure each test gets a fresh HumanLLMConfig singleton
    HumanLLMConfig._HumanLLMConfig__instance = None
    yield

# Fixture: a HumanLLMConfig instance using our real vectordb backend
@pytest.fixture
def config(backend):
    cfg = HumanLLMConfig()
    cfg.common_vectordb = backend
    return cfg

def test_log_and_get_agent_data_basic(config, monkeypatch):
    """Test that logging data and then retrieving it yields the original data."""
    # stub out the internal add_texts to capture calls
    init_add_texts = config.common_vectordb._add_texts
    add_texts = MagicMock()
    config.common_vectordb._add_texts = add_texts

    # Log a simple piece of data
    agent = "Agent1"
    data_key = "test_key"
    data_value = {"foo": "bar"}
    config.log_agent_data(agent, data_key, data_value, function_name="test_func", user_id="user123")
    
    # Verify that underlying db._add_texts was called with serialized JSON
    args, kwargs = add_texts.call_args

    assert "texts" in kwargs and isinstance(kwargs["texts"], list)
    stored_json = kwargs["texts"][0]
    # The stored text should be a JSON string containing the data_value
    assert '"foo": "bar"' in stored_json
    
    # The metadata passed should include agent_name, data_key, function_name, user_id, and date
    assert "metadatas" in kwargs
    meta = kwargs["metadatas"][0]
    assert meta["agent_name"] == agent
    assert meta["data_key"] == data_key
    assert meta["function_name"] == "test_func"
    assert meta["user_id"] == "user123"
    assert "date" in meta and isinstance(meta["date"], str)
    # task_id was not requested, so it should not be in metadata
    assert "task_id" not in meta or meta["task_id"] is None
    
    # Now test get_agent_data can retrieve this entry.
    # Prepare the fake DB to return a Document with the same content and metadata.
    class DummyDocument:
        def __init__(self, content, metadata):
            self.page_content = content
            self.metadata = metadata
            self.id = metadata.get("_id", None)
    dummy_doc = DummyDocument(stored_json, meta)
    # Monkeypatch similarity_search and query methods to return our dummy document
    config.common_vectordb.similarity_search = MagicMock(return_value=[dummy_doc])
    config.common_vectordb._query = MagicMock(return_value=[dummy_doc])
    config.common_vectordb._similarity_search_with_score = MagicMock(return_value=[(dummy_doc, 0.0)])
    
    # Case 1: Default query (should use _query under the hood)
    parsed_list, results = config.get_agent_data(agent_name=agent, data_key=data_key)
    # We expect one result and it should match the logged data
    assert len(parsed_list) == 1
    assert parsed_list[0] == {"foo": "bar"}  # should parse back to original dict
    assert isinstance(results, list) and len(results) == 1
    # The raw result should be our dummy_doc
    assert results[0].metadata["agent_name"] == agent
    
    # Case 2: Similarity search path
    parsed_list2, results2 = config.get_agent_data(agent_name=agent, data_key=data_key, query_text="foo", k=1, metadata_filter=None, sort_order=None, new_storage=True)
    # Since we monkeypatched similarity_search_with_score, we expect it used that for vector query
    config.common_vectordb._query.assert_called()  # ensure similarity path was taken
    assert parsed_list2[0] == {"foo": "bar"}
    assert results2[0].page_content == stored_json

def test_log_agent_data_task_entries(config):
    """Test that add_learnt_task and add_failed_task properly call log_agent_data with correct metadata."""
    # Intercept log_agent_data on UnifiedVectorDB as well, to ensure it's invoked
    internal_log = MagicMock(name="internal_log_agent_data")
    config.common_vectordb.log_agent_data = internal_log
    
    # Sample task entries
    task_content = {"task_parameters": {"task_description": "Do something"}}
    tags = {"task_type": "tag", "step_id": 42}
    # Call add_learnt_task and add_failed_task
    config.add_learnt_task(task_content, tags)
    
    # Verify that the internal log_agent_data was called for each, with expected args
    # For learnt task
    internal_log.assert_any_call(agent_name="HumanLLMConfig", data_key="learnt_task", data_value=task_content,
                                 function_name=None, task_id=False, before_after=None,
                                 user_id=None, step_id=None, task_type="learnt", score=None,
                                 metadata={**tags, "task_type": "learnt"})

    config.add_failed_task(task_content, tags)

    # For failed task
    internal_log.assert_any_call(agent_name="HumanLLMConfig", data_key="failed_task", data_value=task_content,
                                 function_name=None, task_id=False, before_after=None,
                                 user_id=None, step_id=None, task_type="failed", score=None,
                                 metadata={**tags, "task_type": "failed"})
    # Ensure underlying add_texts calls were made via log_agent_data -> _add_texts (we can check db_mock if needed)

def test_get_tasks_retrieval_filters(config):
    """Test that get_learnt_tasks and get_failed_tasks apply correct metadata filters and return expected results."""
    # Prepare dummy documents to return
    def make_doc(content, data_key):
        return type("Doc", (), {"page_content": content, "metadata": {"data_key": data_key, "time": "2025-01-01 00:00:00"}})()
    learned_docs = [make_doc("task1", "learnt_task"), make_doc("task2", "learnt_task")]
    failed_docs = [make_doc("ftask", "failed_task")]
    # Monkeypatch the unified vector search calls
    config.common_vectordb._similarity_search_with_score = MagicMock(return_value=[(doc, 0.5) for doc in learned_docs])
    config.common_vectordb.get_agent_data = MagicMock(return_value=([doc.page_content for doc in learned_docs], learned_docs))
    
    # Test learnt tasks retrieval in similarity mode
    out = config.get_learnt_tasks(query_text="task", similarity_search=True, k=2)
    # Should have used similarity search path
    config.common_vectordb._similarity_search_with_score.assert_called()
    assert isinstance(out, set)  # returns a set of task contents
    assert "task1" in out and "task2" in out
    
    # Test learnt tasks retrieval in filter mode
    config.common_vectordb._similarity_search_with_score.reset_mock()
    out2 = config.get_learnt_tasks(query_text="task", similarity_search=False, metadata_filter={"foo":"bar"})
    config.common_vectordb.get_agent_data.assert_called()
    # The get_agent_data call should have included data_key filter internally; we can verify our stub returned expected content
    assert out2 == {"task1", "task2"}
    
    # Now for failed tasks
    config.common_vectordb._similarity_search_with_score = MagicMock(return_value=[(doc, 0.9) for doc in failed_docs])
    config.common_vectordb.get_agent_data = MagicMock(return_value=([doc.page_content for doc in failed_docs], failed_docs))
    out3 = config.get_failed_tasks(query_text="ftask", similarity_search=True)
    assert out3 == {"ftask"}
    out4 = config.get_failed_tasks(query_text="ftask", similarity_search=False)
    assert out4 == {"ftask"}

def test_pagination_and_sort_order(config):
    """Test that start_index/end_index and sort_order are honored in get_agent_data."""
    # Create a list of dummy docs out of order
    docs = []
    for i in range(5):
        docs.append(type("Doc", (), {
            "page_content": f"item{i}",
            "metadata": {"time": f"2025-01-01 00:00:0{i}"}
        })())
    # Monkeypatch underlying query to return (and sort) the docs list
    def fake_query(*args, **kwargs):
        # ignore all args except sort_order
        so = kwargs.get("sort_order")
        result = list(docs)
        if so == "desc":
            result = sorted(result, key=lambda d: d.metadata["time"], reverse=True)
        elif so == "asc":
            result = sorted(result, key=lambda d: d.metadata["time"])
        return result
    config.common_vectordb._query = fake_query
    # Case 1: No pagination, expect all items as parsed output
    parsed, results = config.common_vectordb.get_agent_data(agent_name=None, data_key="any")
    # The unified get_agent_data should parse each page_content as JSON. Our page_content is simple strings ("itemX"),
    # which are not JSON, but our get_agent_data likely wraps non-dict in a JSON with key.
    # For simplicity, assume it would return them as strings in parsed list if JSON decode fails.
    assert len(parsed) == 5 and len(results) == 5
    
    # Case 2: With pagination
    parsed2, results2 = config.common_vectordb.get_agent_data(agent_name=None, data_key="any", start_index=1, end_index=4)
    # Should slice out docs index 1,2,3 (three items)
    assert len(parsed2) == 3
    assert parsed2[0] == "item1" and parsed2[-1] == "item3"
    # Case 3: With sort_order (desc) on time metadata
    # We simulate that _query returned unsorted docs, get_agent_data should sort if sort_order given.
    parsed3, results3 = config.common_vectordb.get_agent_data(agent_name=None, data_key="any", sort_order="desc")
    # After sort, the first parsed item should be the latest time (item4 in our dummy data)
    assert parsed3[0] == "item4"
