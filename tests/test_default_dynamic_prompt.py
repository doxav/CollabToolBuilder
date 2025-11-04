from types import SimpleNamespace
import pytest

from utils.human_llm import HumanLLM
from utils.human_llm_config import HumanLLMConfig
import utils.llm_utils as llmu


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    def _init_mock(self):
        self.initialized = True
        self.use_websocket = False
        # minimal common_vectordb stub
        self.common_vectordb = SimpleNamespace(
            log_agent_data=lambda *a, **k: None,
            _add_texts=lambda *a, **k: None,
            populate_few_shot_tags=lambda s: s,  # passthrough
        )
    monkeypatch.setattr(HumanLLMConfig, 'initialize', _init_mock, raising=False)
    # Patch both the imported symbol in utils.human_llm and the module-level function
    import utils.human_llm as human_mod
    monkeypatch.setattr(human_mod, 'smart_input', lambda *a, **k: "", raising=False)
    monkeypatch.setattr(llmu, 'smart_input', lambda *a, **k: "", raising=False)


class DummyLLM:
    """Tiny LLM stub mirroring .invoke API and recording the last messages."""
    def __init__(self, name="dummy"):
        self.model_name = name
        self.calls = []
    def with_config(self, **kwargs):
        return self
    def invoke(self, messages, **kwargs):
        # Keep messages to assert prompt composition
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content="[OK]", generation_info={"finish_reason": "stop"})


class EnvStub:
    def __init__(self, state): self._s = state
    def get_state(self): return self._s


def _human_with_dummy(envs=None):
    dummy = DummyLLM()
    return HumanLLM(
        system_prompt="You are a helpful assistant.",
        llmORchains_list={"default_llm": dummy, "premium_llm": dummy},
        num_parallel_inferences=1,
        temperature_min=0.0,
        temperature_max=0.0,
        envs=envs or [],
    ), dummy


def test_dynamic_prompt_strict_placeholders_fill(monkeypatch):
    human, dummy = _human_with_dummy(envs=[EnvStub("ENV_A")])

    # Minimal validation and feedback values via collecting function
    def fake_collect():
        return {
            "env_states": "ENV_A",
            "validation_response_um": "RUNTIME OK",
            "few_shots": "",
            "llm_suggestions": "Try being concise.",
            "llm_annotations": "Consider rephrasing.",
            "llm_feedback_block": "### FEEDBACK SIGNALS\n- Try being concise.\n- Consider rephrasing.",
            "previous_attempts": "",
            "error_patches_str": "",
        }
    monkeypatch.setattr(human, "_collect_compose_values", lambda: fake_collect())

    user_msg = "Env: {env_states}\nRun: {validation_response_um}\nS: {llm_suggestions}\nA: {llm_annotations}\nBlock:\n{llm_feedback_block}"
    out = human.invoke(
        system_prompt_template="You are a helpful assistant.",
        user_message=user_msg,
        stream_output=False,
        return_message_content_only=True,
        compose_mode="strict",
    )
    assert isinstance(out, list)
    assert dummy.calls, "LLM was not invoked"
    last_messages = dummy.calls[-1]["messages"]
    assert len(last_messages) >= 2
    um = last_messages[1].content
    assert "Env: ENV_A" in um
    assert "Run: RUNTIME OK" in um
    assert "S: Try being concise." in um
    assert "A: Consider rephrasing." in um
    assert "### FEEDBACK SIGNALS" in um


def test_dynamic_prompt_auto_appends_context_without_placeholders(monkeypatch):
    human, dummy = _human_with_dummy(envs=[EnvStub("ENV_B")])

    # Patch config agents data -> no feedback yet
    store = {"llm_suggestions": [], "llm_annotations": []}
    monkeypatch.setattr(human.config, "get_agent_data",
                        lambda agent_name=None, data_key=None, **kw:
                        (store.get(data_key, []), store.get(data_key, [])))
    monkeypatch.setattr(human.config, "get_validation_results", lambda: [])

    # Plain prompt, no placeholders: auto-append should include ENVIRONMENT STATE but no FEEDBACK SIGNALS
    human.invoke(
        system_prompt_template="You are a helpful assistant.",
        user_message="Explain memoization.",
        stream_output=False,
        compose_mode="auto",
        return_message_content_only=True,
    )
    last_messages = dummy.calls[-1]["messages"]
    um = last_messages[1].content
    assert "Explain memoization." in um
    assert "### ENVIRONMENT STATE" in um
    assert "ENV_B" in um
    assert "### FEEDBACK SIGNALS" not in um  # no feedback yet


def test_dynamic_prompt_feedback_improves_second_invoke(monkeypatch):
    human, dummy = _human_with_dummy(envs=[EnvStub("ENV_C")])

    # In-memory store for feedback
    store = {"llm_suggestions": [], "llm_annotations": []}
    def fake_log_agent_data(agent_name=None, data_key=None, data_value=None, **kw):
        store.setdefault(data_key, []).append(data_value)
    def fake_get_agent_data(agent_name=None, data_key=None, **kw):
        return (store.get(data_key, []), store.get(data_key, []))
    monkeypatch.setattr(human.config, "log_agent_data", fake_log_agent_data)
    monkeypatch.setattr(human.config, "get_agent_data", fake_get_agent_data)
    monkeypatch.setattr(human.config, "get_validation_results", lambda: [])

    # First call: no feedback present
    human.invoke(
        system_prompt_template="You are a helpful assistant.",
        user_message="Teach recursion simply.",
        stream_output=False,
        compose_mode="auto",
        return_message_content_only=True,
    )
    um1 = dummy.calls[-1]["messages"][1].content
    assert "Teach recursion simply." in um1
    assert "### FEEDBACK SIGNALS" not in um1

    # Provide feedback (simulating a Coach/Critic suggestion recorded)
    human.config.log_agent_data(
        agent_name=human.agent_name,
        data_key="llm_suggestions",
        data_value={"llm_suggestions": "Be more concise and add a small example."}
    )

    # Second call: FEEDBACK SIGNALS block should appear
    human.invoke(
        system_prompt_template="You are a helpful assistant.",
        user_message="Teach recursion simply.",
        stream_output=False,
        compose_mode="auto",
        return_message_content_only=True,
    )
    um2 = dummy.calls[-1]["messages"][1].content
    assert "Teach recursion simply." in um2
    assert "### FEEDBACK SIGNALS" in um2
    assert "Be more concise and add a small example." in um2
