from utils.human_llm import HumanLLM
import pytest

@pytest.fixture(autouse=True)
def _auto_enter(monkeypatch):
    # Avoid interactive prompts in pre_inference by returning Enter/continue
    from utils import llm_utils
    import utils.human_llm as human_llm_mod
    monkeypatch.setattr(llm_utils, 'smart_input', lambda *a, **k: "")
    # Also patch the name imported into human_llm module scope
    if hasattr(human_llm_mod, 'smart_input'):
        monkeypatch.setattr(human_llm_mod, 'smart_input', lambda *a, **k: "")

# Minimal fakes only for these tests

class FakeMessage:
    def __init__(self, content=None, additional_kwargs=None, tool_calls=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}
        self.tool_calls = tool_calls

class FakeModel:
    def __init__(self, returns=None, reject_kwargs=False, name="fake-1"):
        # returns: list of FakeMessage to pop sequentially
        self._queue = list(returns or [])
        self._reject_kwargs = reject_kwargs
        self.model_name = name  # used by code to check temperature hacks

    def invoke(self, messages, **kwargs):
        if self._reject_kwargs and kwargs:
            raise TypeError("Unknown kwargs")
        if self._queue:
            return self._queue.pop(0)
        return FakeMessage(content="")  # default

    def with_config(self, configurable=None):
        return self

    def bind_tools(self, tools):
        # Return a new FakeModel that shares the queue (simulate bound model)
        bound = FakeModel(name=self.model_name)
        bound._queue = self._queue  # share
        bound._reject_kwargs = self._reject_kwargs
        return bound

# Helper to register test tools
def register_test_tool(hllm: HumanLLM, name: str, fn):
    hllm.register_tools({name: fn})

def make_llm(default_model):
    return {"default_llm": default_model, "premium_llm": default_model}

def base_messages():
    return [FakeMessage(content="sys"), FakeMessage(content="user")]

def test_tool_loop_legacy_function_call_path():
    called = {"n": 0}
    def add(x: int, y: int) -> int:
        called["n"] += 1
        return x + y

    # First model output requests a legacy function_call, second returns final content
    m = FakeModel(returns=[
        FakeMessage(content=None, additional_kwargs={"function_call": {"name": "add", "arguments": '{"x":1,"y":2}'}}),
        FakeMessage(content="3")
    ])
    h = HumanLLM(llmORchains_list=make_llm(m))
    register_test_tool(h, "add", add)

    out = h.invoke(original_input_messages=base_messages(), function_calling=True)
    assert isinstance(out, list)
    assert out and out[-1] == "3"
    assert called["n"] == 1

def test_tool_loop_tools_api_path():
    called = {"n": 0}
    def add(x: int, y: int) -> int:
        called["n"] += 1
        return x + y

    m = FakeModel(returns=[
        FakeMessage(content="ignored draft", tool_calls=[{"id": "t1", "type": "function", "function": {"name": "add", "arguments": '{"x":1,"y":2}'}}]),
        FakeMessage(content="ok")
    ])
    h = HumanLLM(llmORchains_list=make_llm(m))
    h.use_tools_api = True  # prefer new tools API
    register_test_tool(h, "add", add)

    out = h.invoke(original_input_messages=base_messages(), function_calling=True)
    assert out and out[-1] == "ok"
    assert called["n"] == 1

def test_tool_loop_mixed_content_and_tool_calls_prefers_tools():
    called = {"n": 0}
    def add(x: int, y: int) -> int:
        called["n"] += 1
        return x + y

    m = FakeModel(returns=[
        FakeMessage(content="draft", tool_calls=[{"id": "t1", "type": "function", "function": {"name": "add", "arguments": '{"x":1,"y":2}'}}]),
        FakeMessage(content="final")
    ])
    h = HumanLLM(llmORchains_list=make_llm(m))
    h.use_tools_api = True
    register_test_tool(h, "add", add)

    out = h.invoke(original_input_messages=base_messages(), function_calling=True)
    assert out and out[-1] == "final"
    assert called["n"] == 1

def test_fallback_when_kwargs_rejected():
    # Model rejects kwargs => tool loop must fall back to plain invoke()
    m = FakeModel(returns=[FakeMessage(content="plain")], reject_kwargs=True)
    h = HumanLLM(llmORchains_list=make_llm(m))
    out = h.invoke(original_input_messages=base_messages(), function_calling=True)
    assert out and out[-1] == "plain"

def test_invoke_calls_tool_loop_when_enabled(monkeypatch):
    m = FakeModel(returns=[
        FakeMessage(content=None, additional_kwargs={"function_call": {"name": "noop", "arguments": "{}"}}),
        FakeMessage(content="done")
    ])
    h = HumanLLM(llmORchains_list=make_llm(m))
    calls = {"n": 0}
    def fake_tool_loop(**kwargs):
        calls["n"] += 1
        # Return second message directly
        return FakeMessage(content="done")
    monkeypatch.setattr(HumanLLM, "_tool_loop", lambda self, **kw: fake_tool_loop(**kw))
    out = h.invoke(original_input_messages=base_messages(), function_calling=True)
    assert calls["n"] == 1
    assert out and out[-1] == "done"

def test_invoke_keeps_non_tool_path_by_default(monkeypatch):
    m = FakeModel(returns=[FakeMessage(content="plain")])
    h = HumanLLM(llmORchains_list=make_llm(m))
    monkeypatch.setattr(HumanLLM, "_tool_loop", lambda self, **kw: (_ for _ in ()).throw(AssertionError("should not be called")))
    out = h.invoke(original_input_messages=base_messages())
    assert out and out[-1] == "plain"

def test_invoke_with_function_call_shim():
    called = {"n": 0}
    def add(x: int, y: int) -> int:
        called["n"] += 1
        return x + y

    # Use wrapper which routes into invoke(tool-mode)
    m = FakeModel(returns=[
        FakeMessage(content=None, additional_kwargs={"function_call": {"name": "add", "arguments": '{"x":2,"y":3}'}}),
        FakeMessage(content="5")
    ])
    h = HumanLLM(llmORchains_list=make_llm(m))
    register_test_tool(h, "add", add)
    out = h.invoke_with_function_call(m, base_messages(), function_list=[
        {"name": "add", "parameters": {"type": "object", "properties": {"x": {"type":"integer"}, "y": {"type":"integer"}}, "required": ["x","y"]}}
    ])
    assert out and out[-1] == "5"
    assert called["n"] == 1
