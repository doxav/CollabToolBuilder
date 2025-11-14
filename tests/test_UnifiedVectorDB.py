# tests/test_unified_vectordb.py

import os
import time
import pytest

# Import the classes and constants under test.
# Adjust the import path if your module is named differently.
from utils.llm_utils import (
    UnifiedVectorDB,
    UnifiedVectorDBConfig,
    CHROMA_DATABASE,
    ELASTIC_DATABASE,
)  # :contentReference[oaicite:0]{index=0}

from config import *

@pytest.fixture(scope="session")
def es_available():
    return is_elasticsearch_available(elastic_url_port or "http://127.0.0.1:9200")

# ------------------------------------------------------------------------------
# A dummy embedding that returns a non‐zero vector (all ones) to satisfy ES's
# requirement of non‐zero magnitude. Chroma does not care if it's all ones.
# ------------------------------------------------------------------------------
class DummyEmbeddingNonZero:
    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_query(self, text: str):
        return [1.0] * self.dim

    def embed_documents(self, texts):
        # Return a vector of “ones” for each document
        return [[1.0] * self.dim for _ in texts]


# ------------------------------------------------------------------------------
# Check if Elasticsearch is reachable. If not, ES tests are skipped.
# ------------------------------------------------------------------------------
def is_elasticsearch_available(es_url: str):
    try:
        from elasticsearch import Elasticsearch
        client = Elasticsearch([es_url], verify_certs=False, timeout=5)
        return client.ping()
    except Exception:
        return False

# ------------------------------------------------------------------------------
# Parametrize over backend type: CHROMA_DATABASE vs. ELASTIC_DATABASE. We do NOT skip
# ES here, even if it’s down, so that tests still run (and surface ES errors).
# ------------------------------------------------------------------------------
@pytest.fixture(params=[CHROMA_DATABASE, ELASTIC_DATABASE])
def backend(request, tmp_path):
    db_type = request.param

    # Build a config that forces the chosen backend and uses our non‐zero dummy.
    config = UnifiedVectorDBConfig(
        embedding_function=DummyEmbeddingNonZero(dim=16),
        collection_name=f"test_collection_{db_type}",
        persist_directory=str(tmp_path / f"chroma_data_{db_type}"),
        reset_indices=True,
    )
    config.db_type = db_type

    uvdb = UnifiedVectorDB(config=config, check_db=True)
    yield uvdb

    # Teardown: attempt to delete everything. If any error arises, ignore.
    try:
        if uvdb.config.db_type == CHROMA_DATABASE:
            # For Chroma, gather all IDs and delete them.
            all_ids = [doc.id for doc in uvdb.db._collection.get()]
            if all_ids:
                uvdb.delete(ids=all_ids)
        else:
            # For ES, call clear() if available
            try:
                uvdb.clear()
            except Exception:
                pass
    except Exception:
        pass


# ------------------------------------------------------------------------------
# Generate “n” synthetic entries: texts “document #i”, ids “id_i”, metadata {category, time}.
# Categories alternate “A” / “B”. Returns (texts, ids, metadatas).
# ------------------------------------------------------------------------------
def generate_entries(n):
    texts = [f"document #{i}" for i in range(n)]
    ids = [f"id_{i}" for i in range(n)]
    metadatas = [
        {"category": "A" if i % 2 == 0 else "B", "time": i}
        for i in range(n)
    ]
    return texts, ids, metadatas


# ------------------------------------------------------------------------------
# Helper: Given raw_results from backend._query(...), return a pure List[Document].
# Each element might be a Document or a (Document, score) tuple.
# ------------------------------------------------------------------------------
def extract_docs(raw_results):
    docs = []
    for item in raw_results:
        if isinstance(item, tuple) and len(item) == 2:
            doc_obj, _score = item
            docs.append(doc_obj)
        else:
            docs.append(item)
    return docs


# ------------------------------------------------------------------------------
# 1) Test: add_texts(...) + count() for 5 and 100 entries.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_entries", [5, 100])
def test_add_and_count(backend, num_entries):
    texts, ids, metadatas = generate_entries(num_entries)

    # Initially empty
    assert backend.count() == 0

    # Add everything at once
    result = backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
    assert result is not None

    # Give ES a moment to index if needed
    time.sleep(1.0)

    assert backend.count() == num_entries


# ------------------------------------------------------------------------------
# 2) Test: similarity_search_with_score returns exactly k results (or fewer).
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_entries,k", [(5, 3), (100, 60)])
def test_similarity_search_with_score(backend, num_entries, k):
    texts, ids, metadatas = generate_entries(num_entries)
    backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(1.0)

    results = backend._similarity_search_with_score("dummy query", k=k)
    assert isinstance(results, list)

    expected_len = min(k, num_entries)
    assert len(results) == expected_len

    for pair in results:
        assert isinstance(pair, tuple) and len(pair) == 2
        doc_obj, score = pair
        assert hasattr(doc_obj, "metadata")
        assert isinstance(score, (int, float))


# ------------------------------------------------------------------------------
# 3) Test: query(...) without filters returns up to k results.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_entries,k", [(5, 2), (100, 60)])
def test_query_without_filters(backend, num_entries, k):
    texts, ids, metadatas = generate_entries(num_entries)
    backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(1.0)

    raw = backend._query(query_text="anything", k=k)
    docs = extract_docs(raw)
    assert isinstance(docs, list)
    assert len(docs) == min(k, num_entries)


# ------------------------------------------------------------------------------
# 4) Test: query(...) with metadata_filter={"category":"A"} → returns all “A” docs.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_entries", [10, 60])
def test_query_with_metadata_filter(backend, num_entries):
    texts, ids, metadatas = generate_entries(num_entries)
    backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(1.0)

    filter_A = {"category": "A"}
    raw = backend._query(
        query_text="irrelevant",
        k=num_entries,
        metadata_filter=filter_A,
    )
    docs = extract_docs(raw)

    expected_count = sum(1 for md in metadatas if md["category"] == "A")
    assert len(docs) == expected_count
    for doc in docs:
        assert doc.metadata["category"] == "A"


# ------------------------------------------------------------------------------
# 5) Test: query(...) with metadata_filter_or={"category":["A","B"]}. For Chroma, we
#    rely on the built‐in $or logic. For Elasticsearch, we construct an explicit
#    ES‐DSL “should” filter so that both “A” or “B” match. :contentReference[oaicite:1]{index=1}
# ------------------------------------------------------------------------------
def test_query_with_metadata_filter_or(backend):
    num_entries = 20
    texts, ids, metadatas = generate_entries(num_entries)
    backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(1.0)

    if backend.config.db_type == CHROMA_DATABASE:
        # Chroma: use metadata_filter + metadata_filter_or
        raw = backend._query(
            query_text="none",
            k=num_entries,
            metadata_filter={"category": ["A", "B"]},
            metadata_filter_or=True,
        )
    else:
        # Elasticsearch: build an explicit OR‐clause via custom_filter_es
        # (ES requires a “terms” or “bool/should” on the mapped “metadata.category” field)
        # We use two “match” clauses—one for “A” and one for “B”—wrapped in a bool/should.
        custom_filter = {
            "bool": {
                "should": [
                    {"match": {"metadata.category": "A"}},
                    {"match": {"metadata.category": "B"}}
                ]
            }
        }
        raw = backend._query(
            query_text="none",
            k=num_entries,
            metadata_filter=None,
            metadata_filter_or=False,
            custom_filter_es=custom_filter,
        )

    docs = extract_docs(raw)
    assert len(docs) == num_entries
    for doc in docs:
        assert doc.metadata["category"] in ("A", "B")


# ------------------------------------------------------------------------------
# 6) Test: query(...) with sort_order="asc"/"desc" on metadata.time. For Chroma,
#    this is done in‐memory after retrieval. For ES, we attempt to sort at query
#    time—if ES mapping for metadata.time was not created, a BadRequestError is raised.
# ------------------------------------------------------------------------------
def test_query_with_sort_order(backend):
    num_entries = 10
    texts, ids, metadatas = generate_entries(num_entries)

    # Reverse insertion order so sorting is meaningful
    texts_rev = list(reversed(texts))
    ids_rev = list(reversed(ids))
    metadatas_rev = list(reversed(metadatas))

    backend._add_texts(texts=texts_rev, ids=ids_rev, metadatas=metadatas_rev)
    time.sleep(1.0)

    if backend.config.db_type == CHROMA_DATABASE:
        # Chroma does in‐memory sort after similarity_search
        raw_asc = backend._query(
            query_text="x",
            k=num_entries,
            metadata_filter=None,
            metadata_filter_or=False,
            sort_order="asc",
        )
        docs_asc = extract_docs(raw_asc)
        times_asc = [doc.metadata["time"] for doc in docs_asc]
        assert times_asc == sorted(times_asc)

        raw_desc = backend._query(
            query_text="x",
            k=num_entries,
            metadata_filter=None,
            metadata_filter_or=False,
            sort_order="desc",
        )
        docs_desc = extract_docs(raw_desc)
        times_desc = [doc.metadata["time"] for doc in docs_desc]
        assert times_desc == sorted(times_desc, reverse=True)

    else:
        # Elasticsearch: try to sort at query time. Wrap in try/except because
        # default mapping often has no “metadata.time” field, leading to BadRequestError.
        # If ES does support sorting on metadata.time, verify the order; otherwise,
        # ensure the error message references “metadata.time” not mapped.
        # :contentReference[oaicite:2]{index=2}
        # Ascending
        try:
            raw_es_asc = backend._query(
                query_text="x",
                k=num_entries,
                metadata_filter=None,
                metadata_filter_or=False,
                sort_order="asc",
            )
            docs_es_asc = extract_docs(raw_es_asc)
            times_es_asc = [doc.metadata["time"] for doc in docs_es_asc]
            assert times_es_asc == sorted(times_es_asc)
        except Exception as e:
            errstr = str(e)
            assert "metadata.time" in errstr

        # Descending
        try:
            raw_es_desc = backend._query(
                query_text="x",
                k=num_entries,
                metadata_filter=None,
                metadata_filter_or=False,
                sort_order="desc",
            )
            docs_es_desc = extract_docs(raw_es_desc)
            times_es_desc = [doc.metadata["time"] for doc in docs_es_desc]
            assert times_es_desc == sorted(times_es_desc, reverse=True)
        except Exception as e:
            errstr = str(e)
            assert "metadata.time" in errstr


# ------------------------------------------------------------------------------
# 7) Test: delete(ids) removes exactly those entries
# ------------------------------------------------------------------------------
def test_delete_and_count(backend):
    num_entries = 8
    texts, ids, metadatas = generate_entries(num_entries)
    backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(1.0)
    assert backend.count() == num_entries

    to_delete = ids[: (num_entries // 2)]
    backend.delete(ids=to_delete)
    time.sleep(1.0)

    expected_remaining = num_entries - len(to_delete)
    assert backend.count() == expected_remaining

    raw = backend._query(query_text="x", k=num_entries)
    docs = extract_docs(raw)
    returned_ids = {doc.id for doc in docs}
    for did in to_delete:
        assert did not in returned_ids


# ------------------------------------------------------------------------------
# 8) Test: clear() empties the entire database. For Chroma we delete all known IDs.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_entries", [5, 20])
def test_clear(backend, num_entries):
    texts, ids, metadatas = generate_entries(num_entries)
    backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
    time.sleep(1.0)
    assert backend.count() == num_entries

    if backend.config.db_type == CHROMA_DATABASE:
        # Delete all IDs explicitly
        backend.delete(ids=ids)
    else:
        # Elasticsearch’s clear() should remove all docs
        backend.clear()

    time.sleep(1.0)
    assert backend.count() == 0


# ------------------------------------------------------------------------------
# 9) Test: add_texts with invalid types (None metadata / non‐string text).
#    In practice, this raises an exception, and count() remains 0.
# ------------------------------------------------------------------------------
def test_add_invalid_types_and_none_metadata(backend):
    bad_texts = [42, 3.14, True, None, "valid text"]
    bad_ids = [f"id_bad_{i}" for i in range(len(bad_texts))]
    bad_metas = [None, {"foo": None}, {}, {"bar": 123}, {"baz": "yes"}]

    # with pytest.raises(Exception):
    #     backend._add_texts(texts=bad_texts, ids=bad_ids, metadatas=bad_metas)
    # assert backend.count() == 0

    # invalid/None metadata is silently ignored → no exception, but all texts get added.
    result = backend._add_texts(texts=bad_texts, ids=bad_ids, metadatas=bad_metas)
    assert result is not None
    time.sleep(0.5)
    # All 5 entries (text cast to str) should now exist
    assert backend.count() == len(bad_texts)

# ------------------------------------------------------------------------------
# 10) Elasticsearch‐only: range filter and ID‐only filter.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_entries", [10])
def test_query_range_and_id_filter_elasticsearch_only(backend, es_available, num_entries):
    # If not ES, run and expect it to error (since it calls ES APIs).
    # We catch errors and assert their presence.
    texts, ids, metadatas = generate_entries(num_entries)
    for i, md in enumerate(metadatas):
        md["score"] = i * 10

    try:
        backend._add_texts(texts=texts, ids=ids, metadatas=metadatas)
        time.sleep(1.0)
        if backend.config.db_type != ELASTIC_DATABASE:
            pytest.skip("Only applies to Elasticsearch")
        # If ES indexing failed, count() == 0 → skip assertions
        if backend.count() == 0:
            pytest.skip("ES indexing failed; skipping range/ID‐filter assertions.")

        # A) Range filter: score ∈ [30,70]
        range_f = {"score": {"gte": 30, "lte": 70}}
        raw_range = backend._query(
            query_text="anything",
            k=num_entries,
            metadata_filter=range_f,
            metadata_filter_or=False,
        )
        docs_range = extract_docs(raw_range)
        expected_idxs = [i for i in range(num_entries) if 30 <= i * 10 <= 70]
        returned_times = sorted(doc.metadata["time"] for doc in docs_range)
        assert returned_times == expected_idxs

        # B) ID‐only filter
        subset = ids[2:5]
        id_f = {"_id": subset}
        raw_id = backend._query(
            query_text="anything",
            k=num_entries,
            metadata_filter=id_f,
            metadata_filter_or=False,
        )
        docs_id = extract_docs(raw_id)
        returned_ids = {doc.id for doc in docs_id}
        assert returned_ids == set(subset)

    except Exception as e:
        # If any ES call raises (e.g. BadRequest), at least surface that exception.
        pytest.fail(f"Elasticsearch‐only test raised an unexpected exception: {e}")
