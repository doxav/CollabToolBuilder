import os
import pytest


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


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    # Ensure config is imported to populate OPENAI_API_KEY if present
    try:
        import config  # noqa: F401
    except Exception:
        pass
    _headless_bootstrap(monkeypatch)


def test_openai_call_with_tools_api(monkeypatch):
    # Gate test execution: require key and RUN_TOOL_LLM_TESTS=1
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_TOOL_LLM_TESTS") == "1"):
        import pytest as _pytest
        _pytest.skip("Set OPENAI_API_KEY and RUN_TOOL_LLM_TESTS=1 to run real LLM test.")

    from utils.human_llm import HumanLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    # Define a simple tool the model could choose to call
    def add(x: int, y: int) -> int:
        return x + y

    tools = [
        {
            "name": "add",
            "description": "Add two integers",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
        }
    ]

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    h = HumanLLM(llmORchains_list={"default_llm": llm, "premium_llm": llm})
    h.use_tools_api = True
    h.register_tools({"add": add})

    messages = [
        SystemMessage(content="You are a helpful assistant that may call tools when useful."),
        HumanMessage(content=(
            "You have a tool 'add' that adds two integers."
            " Call the tool with x=1 and y=2, then reply with 'ok'."
        )),
    ]

    # We don't assert strict tool behavior; we validate the end-to-end call completes
    out = h.invoke(
        original_input_messages=messages,
        function_calling=True,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "add"}},
        max_tool_calls=10,
    )
    # If tool was used, ChatOpenAI typically returns content after a follow-up; accept either path
    assert isinstance(out, list) and len(out) >= 1


def test_openai_tools_forced_choice_calls_tool(monkeypatch):
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_TOOL_LLM_TESTS") == "1"):
        import pytest as _pytest
        _pytest.skip("Set OPENAI_API_KEY and RUN_TOOL_LLM_TESTS=1 to run real LLM test.")

    from utils.human_llm import HumanLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    called = {"flag": False}
    def add(x: int, y: int) -> int:
        called["flag"] = True
        return x + y

    tools = [{
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    }]

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    h = HumanLLM(llmORchains_list={"default_llm": llm, "premium_llm": llm})
    h.use_tools_api = True
    h.register_tools({"add": add})

    messages = [
        SystemMessage(content="You are a helpful assistant that may call tools when useful."),
        HumanMessage(content=("Call the add tool with x=3 and y=4, then reply with 'ok'.")),
    ]

    out = h.invoke(
        original_input_messages=messages,
        function_calling=True,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "add"}},
        max_tool_calls=5,
    )
    assert isinstance(out, list) and len(out) >= 1
    assert called["flag"] is True


def test_openai_tools_auto_may_skip_tool(monkeypatch):
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_TOOL_LLM_TESTS") == "1"):
        import pytest as _pytest
        _pytest.skip("Set OPENAI_API_KEY and RUN_TOOL_LLM_TESTS=1 to run real LLM test.")

    from utils.human_llm import HumanLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    h = HumanLLM(llmORchains_list={"default_llm": llm, "premium_llm": llm})
    h.use_tools_api = True
    # Register a tool but do not force it
    h.register_tools({"noop": lambda: "noop"})

    messages = [
        SystemMessage(content="You are a helpful assistant that may call tools when useful."),
        HumanMessage(content=("Say exactly 'ok' without using any tools.")),
    ]
    out = h.invoke(
        original_input_messages=messages,
        function_calling=True,
        tools=[{"name":"noop","description":"no-op","parameters":{"type":"object","properties":{}}}],
        max_tool_calls=3,
    )
    assert isinstance(out, list) and len(out) >= 1


def test_model_without_bind_tools_fallback(monkeypatch):
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_TOOL_LLM_TESTS") == "1"):
        import pytest as _pytest
        _pytest.skip("Set OPENAI_API_KEY and RUN_TOOL_LLM_TESTS=1 to run real LLM test.")

    from utils.human_llm import HumanLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    class InvokeOnlyWrapper:
        def __init__(self, llm):
            self.llm = llm
            self.model_name = getattr(llm, "model_name", "wrapped")
        def invoke(self, messages):
            return self.llm.invoke(messages)

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    wrapped = InvokeOnlyWrapper(llm)  # no bind_tools, no extra kwargs
    h = HumanLLM(llmORchains_list={"default_llm": wrapped, "premium_llm": wrapped})
    h.use_tools_api = True

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=("Reply 'ok'.")),
    ]
    out = h.invoke(
        original_input_messages=messages,
        function_calling=True,
        tools=[{"name":"noop","description":"no-op","parameters":{"type":"object","properties":{}}}],
        max_tool_calls=2,
    )
    assert isinstance(out, list) and len(out) >= 1


def test_openrouter_if_enabled(monkeypatch):
    # Run only if user explicitly opts in and has base URL pointing to openrouter
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_TOOL_LLM_OPENROUTER") == "1"):
        import pytest as _pytest
        _pytest.skip("Set RUN_TOOL_LLM_OPENROUTER=1 to test via OpenRouter.")
    base = os.environ.get("OPENAI_BASE_URL", "")
    if "openrouter" not in base:
        import pytest as _pytest
        _pytest.skip("OPENAI_BASE_URL does not point to OpenRouter.")

    from utils.human_llm import HumanLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    def add(x: int, y: int) -> int:
        return x + y

    tools = [{
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    }]

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    h = HumanLLM(llmORchains_list={"default_llm": llm, "premium_llm": llm})
    h.use_tools_api = True
    h.register_tools({"add": add})

    messages = [
        SystemMessage(content="You are a helpful assistant that may call tools when useful."),
        HumanMessage(content=("Call the add tool with x=1 and y=2, then reply 'ok'.")),
    ]

    out = h.invoke(
        original_input_messages=messages,
        function_calling=True,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "add"}},
        max_tool_calls=5,
    )
    assert isinstance(out, list) and len(out) >= 1
