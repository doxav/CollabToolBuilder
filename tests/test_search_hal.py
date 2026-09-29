"""Unit tests for the HAL search primitive (network calls are mocked)."""
import importlib.util
import os
from unittest import mock

_PATH = os.path.join(os.path.dirname(__file__), "..", "primitives", "generate_primitives", "search_hal.py")
_spec = importlib.util.spec_from_file_location("search_hal", _PATH)
search_hal_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search_hal_module)
search_hal = search_hal_module.search_hal


def _fake_response(docs):
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": {"numFound": len(docs), "docs": docs}}
    return resp


def test_search_hal_builds_filters_and_parses_docs():
    docs = [{
        "halId_s": "hal-01234567",
        "title_s": ["Réseaux sociaux et violence"],
        "authFullName_s": ["Alice Martin", "Bob Durand"],
        "producedDateY_i": 2023,
        "docType_s": "ART",
        "language_s": ["fr"],
        "abstract_s": ["Résumé de test."],
        "keyword_s": ["réseaux sociaux", "violence"],
        "uri_s": "https://hal.science/hal-01234567",
        "fileMain_s": "https://hal.science/hal-01234567/document",
    }]
    with mock.patch("requests.get", return_value=_fake_response(docs)) as get:
        results = search_hal("réseaux sociaux violence", max_results=5, year_from=2022,
                             doc_types=["ART", "COMM"], language="fr", with_fulltext_only=True)

    params = get.call_args.kwargs["params"]
    assert params["q"] == "réseaux sociaux violence"
    assert params["rows"] == 5
    assert "producedDateY_i:[2022 TO *]" in params["fq"]
    assert "docType_s:(ART OR COMM)" in params["fq"]
    assert "language_s:fr" in params["fq"]
    assert "submitType_s:file" in params["fq"]

    assert results == [{
        "hal_id": "hal-01234567",
        "title": "Réseaux sociaux et violence",
        "authors": "Alice Martin, Bob Durand",
        "year": 2023,
        "doc_type": "ART",
        "language": "fr",
        "abstract": "Résumé de test.",
        "keywords": ["réseaux sociaux", "violence"],
        "url": "https://hal.science/hal-01234567",
        "pdf_url": "https://hal.science/hal-01234567/document",
        "link": "https://hal.science/hal-01234567",
        "description": "Résumé de test.",
    }]


def test_search_hal_handles_missing_fields():
    with mock.patch("requests.get", return_value=_fake_response([{"halId_s": "hal-1"}])):
        (doc,) = search_hal("x")
    assert doc["title"] == "Unknown Title"
    assert doc["abstract"] == "No abstract available"
    assert doc["authors"] == ""
    assert doc["pdf_url"] == ""


def test_search_hal_returns_empty_list_on_http_error():
    with mock.patch("requests.get", side_effect=Exception("boom")):
        assert search_hal("x") == []
