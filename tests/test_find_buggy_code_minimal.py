import types
from primitives.swe_primitives.find_buggy_code import find_buggy_code


class DummyBot:
    def __init__(self):
        self.id = "bot1"

    def getSearchTools(self):
        return []

    def getContentViewingTools(self):
        return []


def test_find_buggy_code_without_env_problem_returns_empty_response():
    bot = DummyBot()
    resp = find_buggy_code(bot)
    assert isinstance(resp, dict)
    assert set(resp.keys()) == {"paths", "line_numbers", "code"}
    assert resp["paths"] == [] and resp["line_numbers"] == [] and resp["code"] == ""

