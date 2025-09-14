import pytest


# --- Headless bootstrap fixture (avoid interactive prompts and external DB writes) ---
@pytest.fixture(autouse=True)
def _headless_bootstrap(monkeypatch):
    from types import SimpleNamespace
    from utils.human_llm_config import HumanLLMConfig
    import utils.llm_utils as llmu

    def _init_mock(self):
        self.initialized = True
        self.use_websocket = False
        self.common_vectordb = SimpleNamespace(log_agent_data=lambda *a, **k: None)

    monkeypatch.setattr(HumanLLMConfig, 'initialize', _init_mock, raising=False)
    monkeypatch.setattr(llmu, 'smart_input', lambda *a, **k: "", raising=False)


# --- Tiny test doubles used by all examples ---
class FakeMessage:
    def __init__(self, content=None, additional_kwargs=None, tool_calls=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}
        self.tool_calls = tool_calls


class FakeModel:
    """Scripted LLM stub that returns queued messages; optionally supports bind_tools."""

    def __init__(self, returns=None, reject_kwargs=False, name="fake-1"):
        self._queue = list(returns or [])
        self._reject_kwargs = reject_kwargs
        self.model_name = name

    def invoke(self, messages, **kwargs):
        if self._reject_kwargs and kwargs:
            raise TypeError("Unknown kwargs")
        if self._queue:
            return self._queue.pop(0)
        return FakeMessage(content="OK")

    def with_config(self, configurable=None):
        return self

    def bind_tools(self, tools):
        bound = FakeModel(name=self.model_name)
        bound._queue = self._queue
        bound._reject_kwargs = self._reject_kwargs
        return bound


def mk_llms(model):
    return {"default_llm": model, "premium_llm": model}


def msgs(sys="sys", user="user"):
    return [FakeMessage(sys), FakeMessage(user)]


# --- Examples mirrored in docs/README_tools.md ---


def test_howto_legacy_functions_path():
    from utils.human_llm import HumanLLM

    calls = {"n": 0}

    def add(x: int, y: int) -> int:
        calls["n"] += 1
        return x + y

    model = FakeModel(
        returns=[
            FakeMessage(
                content=None,
                additional_kwargs={
                    "function_call": {"name": "add", "arguments": '{"x":1,"y":2}'}
                },
            ),
            FakeMessage(content="3"),
        ]
    )

    h = HumanLLM(llmORchains_list=mk_llms(model))
    h.register_tools({"add": add})
    out = h.invoke(original_input_messages=msgs(), function_calling=True)
    assert out[-1] == "3" and calls["n"] == 1


def test_howto_tools_api_bind_tools():
    from utils.human_llm import HumanLLM

    calls = {"n": 0}

    def add(x: int, y: int) -> int:
        calls["n"] += 1
        return x + y

    model = FakeModel(
        returns=[
            FakeMessage(
                content="ignored draft",
                tool_calls=[
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"x":1,"y":2}'},
                    }
                ],
            ),
            FakeMessage(content="ok"),
        ]
    )

    h = HumanLLM(llmORchains_list=mk_llms(model))
    h.use_tools_api = True
    h.register_tools({"add": add})
    out = h.invoke(original_input_messages=msgs(), function_calling=True)
    assert out[-1] == "ok" and calls["n"] == 1


def test_howto_mixed_content_and_tool_calls():
    from utils.human_llm import HumanLLM

    calls = {"n": 0}

    def add(x: int, y: int) -> int:
        calls["n"] += 1
        return x + y

    model = FakeModel(
        returns=[
            FakeMessage(
                content="draft",
                tool_calls=[
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"x":1,"y":2}'},
                    }
                ],
            ),
            FakeMessage(content="final"),
        ]
    )

    h = HumanLLM(llmORchains_list=mk_llms(model))
    h.use_tools_api = True
    h.register_tools({"add": add})
    out = h.invoke(original_input_messages=msgs(), function_calling=True)
    assert out[-1] == "final" and calls["n"] == 1


def test_howto_fallback_when_kwargs_rejected():
    from utils.human_llm import HumanLLM

    model = FakeModel(returns=[FakeMessage(content="plain")], reject_kwargs=True)
    h = HumanLLM(llmORchains_list=mk_llms(model))
    out = h.invoke(original_input_messages=msgs(), function_calling=True)
    assert out[-1] == "plain"


def test_howto_backward_compat_shim():
    from utils.human_llm import HumanLLM

    calls = {"n": 0}

    def add(x: int, y: int) -> int:
        calls["n"] += 1
        return x + y

    model = FakeModel(
        returns=[
            FakeMessage(
                content=None,
                additional_kwargs={
                    "function_call": {"name": "add", "arguments": '{"x":2,"y":3}'}
                },
            ),
            FakeMessage(content="5"),
        ]
    )

    h = HumanLLM(llmORchains_list=mk_llms(model))
    h.register_tools({"add": add})
    out = h.invoke_with_function_call(
        model,
        msgs(),
        function_list=[
            {
                "name": "add",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            }
        ],
    )
    assert out[-1] == "5" and calls["n"] == 1


def test_howto_globals_fallback(monkeypatch):
    from utils.human_llm import HumanLLM
    import utils.human_llm as human_llm_mod

    def multiply(x: int, y: int) -> int:
        return x * y

    # Make function visible to HumanLLM._execute_tool fallback
    monkeypatch.setattr(human_llm_mod, 'multiply', multiply, raising=False)

    model = FakeModel(
        returns=[
            FakeMessage(
                content=None,
                additional_kwargs={
                    "function_call": {"name": "multiply", "arguments": '{"x":3,"y":4}'}
                },
            ),
            FakeMessage(content="12"),
        ]
    )

    h = HumanLLM(llmORchains_list=mk_llms(model))
    out = h.invoke(original_input_messages=msgs(), function_calling=True)
    assert out[-1] == "12"


def test_howto_multi_agent_shared_tool():
    from utils.human_llm import HumanLLM

    def search_for_external_knowledge(description: str, url: str) -> str:
        return f"Searched: {description} at {url}"

    model = FakeModel(returns=[FakeMessage(content="noted")])
    llms = mk_llms(model)

    agents = {
        "planner": HumanLLM(agent_name="Planner", llmORchains_list=llms),
        "coder": HumanLLM(agent_name="Coder", llmORchains_list=llms),
        "tester": HumanLLM(agent_name="Tester", llmORchains_list=llms),
        "reviewer": HumanLLM(agent_name="Reviewer", llmORchains_list=llms),
    }

    for a in agents.values():
        a.register_tools({"search_for_external_knowledge": search_for_external_knowledge})

    out = agents["planner"].invoke(original_input_messages=msgs(), function_calling=True)
    assert out[-1] == "noted"

