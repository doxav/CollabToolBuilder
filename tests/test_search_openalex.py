"""Unit tests for the OpenAlex search primitive (network calls are mocked)."""
import importlib.util
import os
from unittest import mock

_PATH = os.path.join(os.path.dirname(__file__), "..", "primitives", "generate_primitives", "search_arxiv.py")
_spec = importlib.util.spec_from_file_location("search_arxiv_primitives", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
search_openalex = _module.search_openalex


def _fake_response(results):
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"results": results}
    return resp


def test_search_openalex_builds_filters_and_rebuilds_abstract():
    work = {
        "id": "https://openalex.org/W123",
        "title": "Social media and violence",
        "publication_year": 2023,
        "doi": "https://doi.org/10.1234/abc",
        "type": "article",
        "language": "en",
        "cited_by_count": 7,
        "authorships": [{"author": {"display_name": "Alice Martin"}},
                        {"author": {"display_name": "Bob Durand"}}],
        "abstract_inverted_index": {"violence": [2], "Social": [0], "media": [1]},
        "primary_location": {"landing_page_url": "https://hal.science/hal-01234567",
                             "source": {"display_name": "HAL (Le Centre pour la Communication Scientifique Directe)"}},
        "open_access": {"is_oa": True, "oa_url": "https://hal.science/hal-01234567/document"},
    }
    with mock.patch("requests.get", return_value=_fake_response([work])) as get:
        results = search_openalex("social media violence", max_results=500, year_from=2022,
                                  year_to=2024, language="en", open_access_only=True, mailto="a@b.fr")

    params = get.call_args.kwargs["params"]
    assert params["search"] == "social media violence"
    assert params["per_page"] == 200
    assert params["mailto"] == "a@b.fr"
    for f in ["is_paratext:false", "from_publication_date:2022-01-01",
              "to_publication_date:2024-12-31", "language:en", "is_oa:true"]:
        assert f in params["filter"].split(",")

    (doc,) = results
    assert doc["abstract"] == "Social media violence"
    assert doc["year"] == 2023
    assert doc["authors"] == "Alice Martin, Bob Durand"
    assert doc["url"] == "https://hal.science/hal-01234567"
    assert doc["oa_url"] == "https://hal.science/hal-01234567/document"
    assert doc["doi"] == "https://doi.org/10.1234/abc"
    assert doc["source"].startswith("HAL")
    assert doc["openalex_id"] == "https://openalex.org/W123"
    assert doc["link"] == doc["url"]
    assert doc["description"] == doc["abstract"]


def test_search_openalex_handles_missing_fields():
    work = {"id": "https://openalex.org/W1", "primary_location": None, "open_access": None,
            "abstract_inverted_index": None}
    with mock.patch("requests.get", return_value=_fake_response([work])):
        (doc,) = search_openalex("x")
    assert doc["title"] == "Unknown Title"
    assert doc["abstract"] == "No abstract available"
    assert doc["url"] == "https://openalex.org/W1"
    assert doc["source"] == ""


def test_search_openalex_returns_empty_list_on_http_error():
    with mock.patch("requests.get", side_effect=Exception("boom")):
        assert search_openalex("x") == []
