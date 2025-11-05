# tests/test_patch_refine_strategies.py
import types
import difflib
import os
import pathlib
import json
import sys
import pytest

if "config" not in sys.modules:
    config_stub = types.ModuleType("config")
    config_stub.MODELS_CONFIG_LIST = None
    config_stub.embedding_function = None
    config_stub.use_websocket = False
    config_stub.vector_store_type = "chroma"
    config_stub.PickleCacheActivated = False
    config_stub.OPENAI_API_KEY = ""
    config_stub.openai_api_key = ""
    sys.modules["config"] = config_stub

if "jinja2" not in sys.modules:
    jinja2_module = types.ModuleType("jinja2")

    class _StubTemplate:
        def __init__(self, *args, **kwargs):
            pass

        def render(self, *args, **kwargs):
            return ""

    jinja2_module.Template = _StubTemplate
    sys.modules["jinja2"] = jinja2_module

if "matplotlib" not in sys.modules:
    matplotlib_module = types.ModuleType("matplotlib")

    def _stub_rc(*args, **kwargs):
        return None

    matplotlib_module.rc = _stub_rc
    sys.modules["matplotlib"] = matplotlib_module

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class BadRequestError(Exception):
        def __init__(self, *args, param=None, **kwargs):
            super().__init__(*args)
            self.param = param

    openai_module.BadRequestError = BadRequestError
    sys.modules["openai"] = openai_module

if "websockets" not in sys.modules:
    websockets_module = types.ModuleType("websockets")

    async def _stub_connect(*args, **kwargs):  # pragma: no cover - stub only
        class _StubConn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def send(self, *args, **kwargs):
                return None

            async def recv(self, *args, **kwargs):
                return ""

        return _StubConn()

    websockets_module.connect = _stub_connect
    websockets_module.WebSocketException = Exception
    websockets_module.ConnectionClosedError = Exception
    sys.modules["websockets"] = websockets_module

if "requests" not in sys.modules:
    requests_module = types.ModuleType("requests")

    def _stub_request(*args, **kwargs):
        class _Resp:
            status_code = 200

            def json(self):
                return {}

            text = ""

        return _Resp()

    requests_module.get = _stub_request
    requests_module.post = _stub_request
    requests_module.request = _stub_request
    requests_module.Session = type("Session", (), {"__init__": lambda self, *a, **k: None, "request": staticmethod(_stub_request)})

    adapters_module = types.ModuleType("requests.adapters")

    class HTTPAdapter:
        def __init__(self, *args, **kwargs):
            pass

    adapters_module.HTTPAdapter = HTTPAdapter
    sys.modules["requests.adapters"] = adapters_module

    retry_module = types.ModuleType("requests.packages.urllib3.util.retry")

    class Retry:
        def __init__(self, *args, **kwargs):
            pass

    retry_module.Retry = Retry

    sys.modules["requests.packages"] = types.ModuleType("requests.packages")
    sys.modules["requests.packages.urllib3"] = types.ModuleType("requests.packages.urllib3")
    sys.modules["requests.packages.urllib3.util"] = types.ModuleType("requests.packages.urllib3.util")
    sys.modules["requests.packages.urllib3.util.retry"] = retry_module

    auth_module = types.ModuleType("requests.auth")

    class HTTPBasicAuth:
        def __init__(self, *args, **kwargs):
            pass

    auth_module.HTTPBasicAuth = HTTPBasicAuth
    sys.modules["requests.auth"] = auth_module

    requests_module.adapters = adapters_module
    requests_module.packages = types.SimpleNamespace(urllib3=types.SimpleNamespace(util=types.SimpleNamespace(retry=retry_module)))
    requests_module.auth = auth_module
    sys.modules["requests"] = requests_module

if "regex" not in sys.modules:
    import re as _re

    regex_module = types.ModuleType("regex")
    regex_module.compile = _re.compile
    regex_module.Pattern = getattr(_re, "Pattern", type(_re.compile("")))
    regex_module.Match = getattr(_re, "Match", type(_re.match("", "")))
    regex_module.findall = _re.findall
    regex_module.fullmatch = _re.fullmatch
    regex_module.search = _re.search
    regex_module.sub = _re.sub
    sys.modules["regex"] = regex_module

if "sklearn" not in sys.modules:
    sklearn_module = types.ModuleType("sklearn")
    sys.modules["sklearn"] = sklearn_module

    feature_extraction_module = types.ModuleType("sklearn.feature_extraction")
    text_module = types.ModuleType("sklearn.feature_extraction.text")

    class TfidfVectorizer:
        def fit_transform(self, documents):
            return []

        def transform(self, documents):
            return []

    text_module.TfidfVectorizer = TfidfVectorizer
    feature_extraction_module.text = text_module
    sys.modules["sklearn.feature_extraction"] = feature_extraction_module
    sys.modules["sklearn.feature_extraction.text"] = text_module

    metrics_module = types.ModuleType("sklearn.metrics")
    pairwise_module = types.ModuleType("sklearn.metrics.pairwise")

    def cosine_similarity(*args, **kwargs):
        return []

    pairwise_module.cosine_similarity = cosine_similarity
    metrics_module.pairwise = pairwise_module
    sys.modules["sklearn.metrics"] = metrics_module
    sys.modules["sklearn.metrics.pairwise"] = pairwise_module

if "langchain_openai" not in sys.modules:
    lc_openai = types.ModuleType("langchain_openai")

    class _StubChat:
        model_name = "stub"

        def __init__(self, *args, **kwargs):
            pass

        def with_config(self, configurable=None):
            return self

        def invoke(self, messages, **kwargs):
            return types.SimpleNamespace(content="")

    lc_openai.ChatOpenAI = _StubChat
    lc_openai.AzureChatOpenAI = _StubChat
    sys.modules["langchain_openai"] = lc_openai
else:
    lc_openai = sys.modules["langchain_openai"]

ART_DIR = pathlib.Path(__file__).resolve().parent.parent / "test_artifacts"

if "langchain_community" not in sys.modules:
    lc_community = types.ModuleType("langchain_community")
    lc_community.__path__ = []
    sys.modules["langchain_community"] = lc_community
else:
    lc_community = sys.modules["langchain_community"]

if "langchain_community.embeddings" not in sys.modules:
    embeddings_module = types.ModuleType("langchain_community.embeddings")

    class _StubEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, texts):
            return [[0.0] * len(texts) for _ in texts]

        def embed_query(self, text):
            return []

    embeddings_module.OpenAIEmbeddings = _StubEmbedding
    embeddings_module.HuggingFaceEmbeddings = _StubEmbedding
    lc_community.embeddings = embeddings_module
    sys.modules["langchain_community.embeddings"] = embeddings_module

if "langchain_community.vectorstores" not in sys.modules:
    vectorstores_module = types.ModuleType("langchain_community.vectorstores")

    class _StubVectorStore:
        def __init__(self, *args, **kwargs):
            pass

    vectorstores_module.ElasticsearchStore = _StubVectorStore
    lc_community.vectorstores = vectorstores_module
    sys.modules["langchain_community.vectorstores"] = vectorstores_module

if "langchain_community.chat_models" not in sys.modules:
    chat_models_module = types.ModuleType("langchain_community.chat_models")

    class _StubChatModel:
        def __init__(self, *args, **kwargs):
            pass

    chat_models_module.ChatOllama = _StubChatModel
    lc_community.chat_models = chat_models_module
    sys.modules["langchain_community.chat_models"] = chat_models_module

if "langchain" in sys.modules:
    langchain_module = sys.modules["langchain"]
else:
    langchain_module = types.ModuleType("langchain")
    sys.modules["langchain"] = langchain_module
if not hasattr(langchain_module, "__path__"):
    langchain_module.__path__ = []

if "langchain.schema" not in sys.modules:
    schema_module = types.ModuleType("langchain.schema")

    class Document:
        def __init__(self, page_content: str = "", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema_module.Document = Document
    langchain_module.schema = schema_module
    sys.modules["langchain.schema"] = schema_module

if "langchain.llms" not in sys.modules:
    llms_module = types.ModuleType("langchain.llms")

    class _StubLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages, **kwargs):
            return types.SimpleNamespace(content="")

    llms_module.OpenAI = _StubLLM
    langchain_module.llms = llms_module
    sys.modules["langchain.llms"] = llms_module

if "langchain.chains" not in sys.modules:
    chains_module = types.ModuleType("langchain.chains")

    class _StubChain:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return ""

    chains_module.LLMChain = _StubChain
    langchain_module.chains = chains_module
    sys.modules["langchain.chains"] = chains_module

if "PyPDF2" not in sys.modules:
    pypdf2_module = types.ModuleType("PyPDF2")
    generic_module = types.ModuleType("PyPDF2.generic")

    class IndirectObject:
        def __init__(self, *args, **kwargs):
            pass

    generic_module.IndirectObject = IndirectObject
    pypdf2_module.generic = generic_module
    sys.modules["PyPDF2"] = pypdf2_module
    sys.modules["PyPDF2.generic"] = generic_module

if "langchain.prompts" not in sys.modules:
    prompts_module = types.ModuleType("langchain.prompts")

    class PromptTemplate:
        def __init__(self, *args, **kwargs):
            pass

        def format(self, **kwargs):
            return ""

    prompts_module.PromptTemplate = PromptTemplate
    langchain_module.prompts = prompts_module
    sys.modules["langchain.prompts"] = prompts_module

if "opentelemetry" not in sys.modules:
    otel_module = types.ModuleType("opentelemetry")
    sys.modules["opentelemetry"] = otel_module

if "opentelemetry.trace" not in sys.modules:
    otel_trace_module = types.ModuleType("opentelemetry.trace")

    class _StubSpan:
        def set_attribute(self, key, value):
            pass

        def is_recording(self):
            return False

    def _get_tracer(*args, **kwargs):
        class _Tracer:
            def start_as_current_span(self, *args, **kwargs):
                from contextlib import contextmanager

                @contextmanager
                def _cm():
                    yield _StubSpan()

                return _cm()

        return _Tracer()

    def _get_current_span():
        return _StubSpan()

    otel_trace_module.get_tracer = _get_tracer
    otel_trace_module.get_current_span = _get_current_span
    otel_trace_module.Span = _StubSpan
    sys.modules["opentelemetry.trace"] = otel_trace_module
    sys.modules["opentelemetry"].trace = otel_trace_module

if "langchain_core" not in sys.modules:
    langchain_core_module = types.ModuleType("langchain_core")
    sys.modules["langchain_core"] = langchain_core_module
else:
    langchain_core_module = sys.modules["langchain_core"]
if not hasattr(langchain_core_module, "__path__"):
    langchain_core_module.__path__ = []

messages_module = types.ModuleType("langchain_core.messages")


class _BaseMessage:
    def __init__(self, content=""):
        self.content = content


messages_module.ai = types.ModuleType("langchain_core.messages.ai")
messages_module.human = types.ModuleType("langchain_core.messages.human")
messages_module.system = types.ModuleType("langchain_core.messages.system")
messages_module.function = types.ModuleType("langchain_core.messages.function")


class AIMessage(_BaseMessage):
    pass


class HumanMessage(_BaseMessage):
    pass


class SystemMessage(_BaseMessage):
    pass


class FunctionMessage(_BaseMessage):
    pass


messages_module.ai.AIMessage = AIMessage
messages_module.human.HumanMessage = HumanMessage
messages_module.system.SystemMessage = SystemMessage
messages_module.function.FunctionMessage = FunctionMessage

sys.modules["langchain_core.messages"] = messages_module
sys.modules["langchain_core.messages.ai"] = messages_module.ai
sys.modules["langchain_core.messages.human"] = messages_module.human
sys.modules["langchain_core.messages.system"] = messages_module.system
sys.modules["langchain_core.messages.function"] = messages_module.function
langchain_core_module.messages = messages_module

runnables_module = types.ModuleType("langchain_core.runnables")


class Runnable:
    def __init__(self, *args, **kwargs):
        pass


class RunnableSequence(Runnable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConfigurableField:
    def __init__(self, *args, **kwargs):
        pass


runnables_module.Runnable = Runnable
runnables_module.RunnableSequence = RunnableSequence
runnables_module.ConfigurableField = ConfigurableField
sys.modules["langchain_core.runnables"] = runnables_module
langchain_core_module.runnables = runnables_module

prompts_module = types.ModuleType("langchain_core.prompts")


class ChatPromptTemplate:
    @classmethod
    def from_messages(cls, messages):
        return cls()

    def format(self, **kwargs):
        return ""


prompts_module.ChatPromptTemplate = ChatPromptTemplate
sys.modules["langchain_core.prompts"] = prompts_module
langchain_core_module.prompts = prompts_module

from langchain_core.messages.ai import AIMessage

from utils.human_llm import HumanLLM
from utils.human_llm_config import HumanLLMConfig

HumanLLMConfig.initialize = lambda self: None


class _DummyLLM:
    """Minimal drop-in for default/premium_llm used by HumanLLM."""

    model_name = "dummy"

    def with_config(self, configurable=None):
        return self

    def invoke(self, messages, **kwargs):
        return types.SimpleNamespace(content="(dummy)")


def _llm_pool():
    return {"default_llm": _DummyLLM(), "premium_llm": _DummyLLM()}


def _make_human_llm(dynamic_cfg=None):
    return HumanLLM(
        agent_name="PatchRefineTest",
        llmORchains_list=_llm_pool(),
        num_parallel_inferences=2,
        dynamic_llm_config=dynamic_cfg or {}
    )


def _unified_diff(old_text: str, new_text: str, filename: str = "answer.md") -> str:
    old = (old_text or "").splitlines(keepends=True)
    new = (new_text or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        old,
        new,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    )
    joined = "\n".join(diff)
    if joined and not joined.endswith("\n"):
        joined += "\n"
    return joined


def _draft_msg(text: str) -> AIMessage:
    return AIMessage(content=f"DR AFT\n{text}")


def _patch_msg_from_diff(diff: str) -> AIMessage:
    return AIMessage(content=diff)


def _patch_msg_from_json_ops(ops: list[dict]) -> AIMessage:
    return AIMessage(content="JSON_PATCH\n" + json.dumps(ops))


@pytest.fixture
def dyncfg_multi_agent_vote():
    return {
        "multi_agent_patch_vote": {
            "rules": {"regex": {"patterns": [r"multi-agent"], "target": "user_message"}},
            "modifications": {
                "generation_technique": "draft_then_patch",
                "selection_technique": "patch_hunk_vote",
                "num_parallel_inferences": 4
            }
        }
    }


@pytest.fixture
def dyncfg_best_of_n():
    return {
        "best_of_n_refine": {
            "rules": {"regex": {"patterns": [r"best-of-n"], "target": "user_message"}},
            "modifications": {
                "generation_technique": "draft_then_patch",
                "selection_technique": "patch_best_of_n",
                "num_parallel_inferences": 6
            }
        }
    }


def test_dynamic_config_switches_to_hunk_vote(dyncfg_multi_agent_vote):
    h = _make_human_llm(dynamic_cfg=dyncfg_multi_agent_vote)
    ctx = {"user_message": "Please use multi-agent refinement."}
    mods = h.dynamic_mgr.evaluate_triggers(ctx, phase="pre_inference")
    h._apply_modifications(mods, ctx, "pre_inference")
    assert h.generation_technique == "draft_then_patch"
    assert h.selection_technique == "patch_hunk_vote"
    assert h.num_parallel_inferences == 4


def test_dynamic_config_switches_to_best_of_n(dyncfg_best_of_n):
    h = _make_human_llm(dynamic_cfg=dyncfg_best_of_n)
    ctx = {"user_message": "Use best-of-n refinement."}
    mods = h.dynamic_mgr.evaluate_triggers(ctx, phase="pre_inference")
    h._apply_modifications(mods, ctx, "pre_inference")
    assert h.generation_technique == "draft_then_patch"
    assert h.selection_technique == "patch_best_of_n"
    assert h.num_parallel_inferences == 6


def test_patch_hunk_vote_merges_non_conflicting_hunks():
    baseline = "# Plan\n- Alpha\n- Beta\n- Gamma\n"
    pA = _unified_diff(
        baseline,
        "# Plan\n- Alpha\n- Beta (improved)\n- Gamma\n"
    )
    pB = _unified_diff(
        baseline,
        "# Plan\n- Alpha\n- Beta\n- Gamma (revised)\n"
    )
    pC = _unified_diff(
        baseline,
        "# Plan\n- Alpha\n- Beta (improved)\n- Gamma\n"
    )

    h = _make_human_llm()
    outs = [
        _draft_msg(baseline),
        _patch_msg_from_diff(pA),
        _patch_msg_from_diff(pB),
        _patch_msg_from_diff(pC),
    ]
    final = h.select_candidate(outs, selection_technique="patch_hunk_vote")[0].content
    assert "- Beta (improved)" in final
    assert "- Gamma (revised)" in final
    assert final.startswith("# Plan")


def test_patch_best_of_n_picks_consensus():
    baseline = "X\nY\nZ\n"
    p1 = _unified_diff(baseline, "X\nY*\nZ\n")
    p2 = _unified_diff(baseline, "X\nY*\nZ\n")
    p3 = _unified_diff(baseline, "X\nY2\nZ\n")

    h = _make_human_llm()
    outs = [_draft_msg(baseline), _patch_msg_from_diff(p1), _patch_msg_from_diff(p2), _patch_msg_from_diff(p3)]
    final = h.select_candidate(outs, selection_technique="patch_best_of_n")[0].content
    assert "Y*" in final and "Y2" not in final


def test_json_ops_edits_are_supported_in_hunk_vote():
    baseline = "# Title\n## Budget\nTBD\n## Grid\nOld\n"
    json_ops = [
        {"op": "replace", "loc": {"type": "section", "title": "Budget"}, "text": "Total: €10M"}
    ]
    p_json = _patch_msg_from_json_ops(json_ops)
    p_diff = _patch_msg_from_diff(_unified_diff(baseline, "# Title\n## Budget\nTBD\n## Grid\nNew Grid Plan\n"))

    h = _make_human_llm()
    outs = [_draft_msg(baseline), p_json, p_diff]
    final = h.select_candidate(outs, selection_technique="patch_hunk_vote")[0].content
    assert "## Budget\nTotal: €10M" in final
    assert "## Grid\nNew Grid Plan" in final


def test_selector_is_robust_to_invalid_patches_and_mixed_inputs():
    baseline = "A\nB\nC\n"
    invalid = AIMessage(content="<<<NOT A DIFF>>>")
    full = AIMessage(content="FULL\nA\nB*\nC\n")

    h = _make_human_llm()
    outs = [_draft_msg(baseline), invalid, full]
    final = h.select_candidate(outs, selection_technique="patch_hunk_vote")[0].content
    assert "B*" in final and "<<<" not in final


def test_draft_patch_mode_generates_draft_and_diffs(monkeypatch):
    h = _make_human_llm()
    h.draft_patch_mode = True

    baseline = "X\nY\nZ\n"
    rewrites = ["X\nY*\nZ\n", "X\nY2\nZ\n"]

    monkeypatch.setattr(h, "_draft_patch_baseline", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(
        h,
        "_draft_patch_rewrites",
        lambda *args, **kwargs: [AIMessage(content=s) for s in rewrites],
    )

    msgs = h.generate_candidates(
        generation_technique="temperature_variation",
        system_prompt="You are helpful.",
        user_prompt="Draft a plan, then refine it.",
        num_responses=2,
        use_premium_llm=False,
        function_calling=False,
        temp_min=0.1,
        temp_max=0.3,
        stream_output=False,
    )

    assert isinstance(msgs, list)
    assert msgs[0].content.startswith("DR AFT\n")
    assert any(m.content.startswith("--- a/") or "@@" in m.content for m in msgs[1:])


# ---------------------------------------------------------------------------
# Integration tests (require OPENAI_API_KEY)
# ---------------------------------------------------------------------------

_CHAT_OPENAI = getattr(lc_openai, "ChatOpenAI", None)
_is_stub_chat = getattr(_CHAT_OPENAI, "__module__", "") == __name__

requires_key = pytest.mark.skipif(
    (not os.environ.get("OPENAI_API_KEY")) or _is_stub_chat,
    reason="OPENAI_API_KEY not set or langchain_openai unavailable; skipping live LLM tests.",
)


def _ensure_art_dir():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    return ART_DIR


@requires_key
def test_llm_code_patch_executes_and_changes(tmp_path):
    from langchain_openai import ChatOpenAI
    from utils.human_llm import HumanLLM

    _ensure_art_dir()

    llm_pool = {
        "default_llm": ChatOpenAI(model="gpt-5-nano", temperature=0.0),
        "premium_llm": ChatOpenAI(model="gpt-5-nano", temperature=0.0),
    }

    h = HumanLLM(
        agent_name="IT-test-code",
        llmORchains_list=llm_pool,
        num_parallel_inferences=2,
        dynamic_llm_config={},
    )
    h.draft_patch_mode = True

    system_prompt = "You are a helpful Python assistant. Keep unchanged lines identical when refining."
    user_prompt = (
        "Write a short Python script named answer.py that computes factorial(6) and prints the result.\n"
        "Then, when refining, make small improvements like adding a helper or an extra print, but preserve structure."
    )

    msgs = h.generate_candidates(
        generation_technique="temperature_variation",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_responses=2,
        use_premium_llm=False,
        function_calling=False,
        temp_min=0.1,
        temp_max=0.3,
        stream_output=False,
    )

    (ART_DIR / "code_draft_and_patches.json").write_text(
        json.dumps([m.content for m in msgs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline = next((m.content.split("\n", 1)[1] for m in msgs if m.content.startswith("DR AFT")), "")
    assert baseline.strip()

    final_msg = h.select_candidate(msgs, selection_technique="patch_hunk_vote")[0]
    final = final_msg.content
    (ART_DIR / "code_final_merged.py").write_text(final, encoding="utf-8")

    assert final != baseline
    compiled = compile(final, "<merged>", "exec")
    namespace = {}
    exec(compiled, namespace, namespace)


@requires_key
def test_llm_json_patch_is_valid_and_differs(tmp_path):
    from langchain_openai import ChatOpenAI
    from utils.human_llm import HumanLLM

    _ensure_art_dir()

    llm_pool = {
        "default_llm": ChatOpenAI(model="gpt-5-nano", temperature=0.0),
        "premium_llm": ChatOpenAI(model="gpt-5-nano", temperature=0.0),
    }

    h = HumanLLM(
        agent_name="IT-test-json",
        llmORchains_list=llm_pool,
        num_parallel_inferences=2,
        dynamic_llm_config={},
    )
    h.draft_patch_mode = True

    system_prompt = "You are a structured editor. Prefer JSON_PATCH with section-level replace ops when asked."
    user_prompt = (
        "Create a Markdown plan with sections (# Title, ## Budget, ## Steps). "
        "Then refine it by proposing JSON_PATCH edits that change the Budget and add a detail to Steps."
    )

    msgs = h.generate_candidates(
        generation_technique="temperature_variation",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_responses=2,
        use_premium_llm=True,
        function_calling=False,
        temp_min=0.1,
        temp_max=0.2,
        stream_output=False,
    )

    (ART_DIR / "jsonpatch_draft_and_patches.json").write_text(
        json.dumps([m.content for m in msgs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline = next((m.content.split("\n", 1)[1] for m in msgs if m.content.startswith("DR AFT")), "")
    assert baseline.strip()

    json_msgs = [m for m in msgs if m.content.upper().startswith("JSON_PATCH")]
    assert json_msgs
    payload = json.loads(json_msgs[0].content.split("\n", 1)[1])
    assert isinstance(payload, list)

    final = h.select_candidate(msgs, selection_technique="patch_hunk_vote")[0].content
    (ART_DIR / "jsonpatch_final_merged.md").write_text(final, encoding="utf-8")
    assert final != baseline
    assert "## Budget" in final
    assert "## Steps" in final


@requires_key
def test_llm_text_patch_is_longer_after_merge(tmp_path):
    from langchain_openai import ChatOpenAI
    from utils.human_llm import HumanLLM

    _ensure_art_dir()

    llm_pool = {
        "default_llm": ChatOpenAI(model="gpt-5-nano", temperature=0.0),
        "premium_llm": ChatOpenAI(model="gpt-5-nano", temperature=0.0),
    }

    h = HumanLLM(
        agent_name="IT-test-text",
        llmORchains_list=llm_pool,
        num_parallel_inferences=2,
        dynamic_llm_config={},
    )
    h.draft_patch_mode = True

    system_prompt = "You expand content cautiously. Keep unchanged lines identical where possible."
    user_prompt = "Write a 3-4 line description of a morning routine. Then refine to add 2 extra specific tips."

    msgs = h.generate_candidates(
        generation_technique="temperature_variation",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_responses=2,
        use_premium_llm=False,
        function_calling=False,
        temp_min=0.2,
        temp_max=0.4,
        stream_output=False,
    )

    (ART_DIR / "text_draft_and_patches.json").write_text(
        json.dumps([m.content for m in msgs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline = next((m.content.split("\n", 1)[1] for m in msgs if m.content.startswith("DR AFT")), "")
    final = h.select_candidate(msgs, selection_technique="patch_hunk_vote")[0].content
    (ART_DIR / "text_final_merged.md").write_text(final, encoding="utf-8")
    assert len(final) > len(baseline)
