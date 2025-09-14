# Tools: Building and Using Them with HumanLLM

This guide shows, by code, how to:
- Build and register tools for agents (learn.py / CollabToolBuilder).
- Use tools through HumanLLM with both OpenAI’s legacy “functions” and the newer “tools” API.

All examples are headless (no network or DB required), using the same tiny fakes we use in unit tests.

## Minimal headless bootstrap

Disable interactive prompts and vector DB writes for quick demos.

```python
from types import SimpleNamespace
from utils.human_llm_config import HumanLLMConfig
import utils.llm_utils as llmu

def _init_mock(self):
    self.initialized = True
    self.use_websocket = False
    self.common_vectordb = SimpleNamespace(log_agent_data=lambda *a, **k: None)

HumanLLMConfig.initialize = _init_mock
llmu.smart_input = lambda *a, **k: ""
```

## Tiny test doubles (used in all examples)

```python
class FakeMessage:
    def __init__(self, content=None, additional_kwargs=None, tool_calls=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}
        self.tool_calls = tool_calls

class FakeModel:
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
```

---

## Using tools in HumanLLM

HumanLLM.invoke supports both legacy OpenAI “functions” and the newer “tools” (tool_calls) paths, selectable at call time. Tool loop is OFF by default; enable with `function_calling=True` (or by passing `tools` / `functions`). Streaming remains unchanged (tool loop runs only when `stream_output=False`).

### 1) Legacy “functions” path

```python
from utils.human_llm import HumanLLM

calls = {"n": 0}
def add(x: int, y: int) -> int:
    calls["n"] += 1
    return x + y

model = FakeModel(returns=[
    FakeMessage(content=None, additional_kwargs={
        "function_call": {"name": "add", "arguments": '{"x":1,"y":2}'}
    }),
    FakeMessage(content="3"),
])

h = HumanLLM(llmORchains_list=mk_llms(model))
h.register_tools({"add": add})
out = h.invoke(original_input_messages=msgs(), function_calling=True)
assert out[-1] == "3" and calls["n"] == 1
```

### 2) New “tools” path (bind_tools)

```python
calls = {"n": 0}
def add(x: int, y: int) -> int:
    calls["n"] += 1
    return x + y

model = FakeModel(returns=[
    FakeMessage(content="ignored draft", tool_calls=[{
        "id": "t1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"x":1,"y":2}'}
    }]),
    FakeMessage(content="ok"),
])

h = HumanLLM(llmORchains_list=mk_llms(model))
h.use_tools_api = True
h.register_tools({"add": add})
out = h.invoke(
    original_input_messages=msgs(),
    function_calling=True,
    tools=[{"name":"add","description":"Add two integers","parameters":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"]}}],
    tool_choice={"type":"function","function":{"name":"add"}},  # force tool usage when supported
    max_tool_calls=8,
)
assert out[-1] == "ok" and calls["n"] == 1
```

### 3) Mixed content + tool_calls

```python
calls = {"n": 0}
def add(x: int, y: int) -> int:
    calls["n"] += 1
    return x + y

model = FakeModel(returns=[
    FakeMessage(content="draft", tool_calls=[{
        "id": "t1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"x":1,"y":2}'}
    }]),
    FakeMessage(content="final"),
])

h = HumanLLM(llmORchains_list=mk_llms(model))
h.use_tools_api = True
h.register_tools({"add": add})
out = h.invoke(original_input_messages=msgs(), function_calling=True)
assert out[-1] == "final" and calls["n"] == 1
```

### 4) Fallback when chain rejects unknown kwargs

```python
model = FakeModel(returns=[FakeMessage(content="plain")], reject_kwargs=True)
h = HumanLLM(llmORchains_list=mk_llms(model))
out = h.invoke(original_input_messages=msgs(), function_calling=True)
assert out[-1] == "plain"
```

### 5) Backward‑compat shim

```python
calls = {"n": 0}
def add(x: int, y: int) -> int:
    calls["n"] += 1
    return x + y

model = FakeModel(returns=[
    FakeMessage(content=None, additional_kwargs={
        "function_call": {"name": "add", "arguments": '{"x":2,"y":3}'}
    }),
    FakeMessage(content="5"),
])

h = HumanLLM(llmORchains_list=mk_llms(model))
h.register_tools({"add": add})
out = h.invoke_with_function_call(model, msgs(), function_list=[
    {"name": "add", "parameters": {"type": "object", "properties": {"x": {"type":"integer"}, "y": {"type":"integer"}}, "required": ["x","y"]}}
])
assert out[-1] == "5" and calls["n"] == 1
```

### 6) Optional globals() fallback

```python
import utils.human_llm as human_llm_mod

def multiply(x:int, y:int) -> int:
    return x*y

human_llm_mod.multiply = multiply  # visible to HumanLLM._execute_tool()

model = FakeModel(returns=[
    FakeMessage(content=None, additional_kwargs={
        "function_call": {"name": "multiply", "arguments": '{"x":3,"y":4}'}
    }),
    FakeMessage(content="12"),
])
h = HumanLLM(llmORchains_list=mk_llms(model))
out = h.invoke(original_input_messages=msgs(), function_calling=True)
assert out[-1] == "12"
```

---

## Building tools with CollabToolBuilder’s agents (4‑agent pattern)

```python
from utils.human_llm import HumanLLM

def search_for_external_knowledge(description: str, url: str) -> str:
    return f"Searched: {description} at {url}"

model = FakeModel(returns=[FakeMessage(content="noted")])
llms = mk_llms(model)

agents = {
    "planner":  HumanLLM(agent_name="Planner",  llmORchains_list=llms),
    "coder":    HumanLLM(agent_name="Coder",    llmORchains_list=llms),
    "tester":   HumanLLM(agent_name="Tester",   llmORchains_list=llms),
    "reviewer": HumanLLM(agent_name="Reviewer", llmORchains_list=llms),
}

for a in agents.values():
    a.register_tools({"search_for_external_knowledge": search_for_external_knowledge})

out = agents["planner"].invoke(original_input_messages=msgs(), function_calling=True)
assert out[-1] == "noted"
```

---

## Parameter cheatsheet (HumanLLM.invoke)

- `function_calling`: Master ON/OFF for the tool loop (non‑streaming).
- `use_tools_api`: Prefer `bind_tools()` when True (default False for full backward compatibility).
- `tools`: New tools API descriptors.
- `functions`: Legacy OpenAI function descriptors.
- `tool_choice` / `function_call`: Typically "auto".
- `max_tool_calls`: Max tool steps (default 5).

Notes:
- Streaming is unchanged; tool loop runs only when `stream_output=False`.
- Prefer `register_tools` (safe whitelist). Fallback to module globals exists for legacy compatibility.

---

## Real LLM (optional)

You can run an end‑to‑end test against OpenAI models if you set these environment variables:

```bash
export OPENAI_API_KEY=sk-...
export RUN_TOOL_LLM_TESTS=1
pytest -q tests/test_tools_openai_optional.py -q
```

Tip: To force the model to call a specific tool, pass `tool_choice={"type":"function","function":{"name":"<tool>"}}` along with a proper `tools` schema, as shown above.

This optional test uses `ChatOpenAI` with `use_tools_api=True` and validates that a full invocation completes without errors. It is skipped by default to keep CI deterministic and offline.
