import importlib
from types import SimpleNamespace

import pytest


class DummyVectorConfig:
    def __init__(self):
        self.embedding_function = "initial"
        self.reset_indices = False
        self.unique_collection_id = None

    def set_unique_collection_id(self, unique_id):
        self.unique_collection_id = unique_id


class DummyHumanLLMConfig:
    def __init__(self):
        self.use_websocket = False
        self.smart_input = None
        self.smart_print = None
        self.ws_server_config = SimpleNamespace(port=None, secret=None, proxy_enabled=None)
        self.common_vectordb_config = DummyVectorConfig()
        self.initialized = False

    def initialize(self):
        self.initialized = True


@pytest.fixture(autouse=True)
def reset_embedding_singleton(monkeypatch):
    import sys
    import types

    jinja_stub = types.ModuleType("jinja2")
    jinja_stub.Template = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "jinja2", jinja_stub)

    class _DummyEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def embed_query(self, text):
            return [1.0]

        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

    embeddings_stub = types.ModuleType("langchain_community.embeddings")
    embeddings_stub.OpenAIEmbeddings = _DummyEmbedding
    embeddings_stub.HuggingFaceEmbeddings = _DummyEmbedding

    langchain_community_stub = types.ModuleType("langchain_community")
    langchain_community_stub.embeddings = embeddings_stub

    monkeypatch.setitem(sys.modules, "langchain_community", langchain_community_stub)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", embeddings_stub)

    schema_stub = types.ModuleType("langchain.schema")

    class _DummyDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema_stub.Document = _DummyDocument
    monkeypatch.setitem(sys.modules, "langchain.schema", schema_stub)

    requests_stub = types.ModuleType("requests")

    class _DummySession:
        def __init__(self, *args, **kwargs):
            pass

        def mount(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            class _DummyResponse:
                status_code = 200
                text = ""

                def json(self):
                    return {}

            return _DummyResponse()

    class _RequestException(Exception):
        pass

    requests_stub.Session = _DummySession
    requests_stub.exceptions = types.SimpleNamespace(RequestException=_RequestException)

    requests_adapters_stub = types.ModuleType("requests.adapters")

    class _HTTPAdapter:
        def __init__(self, *args, **kwargs):
            pass

    requests_adapters_stub.HTTPAdapter = _HTTPAdapter

    requests_auth_stub = types.ModuleType("requests.auth")

    class _HTTPBasicAuth:
        def __init__(self, *args, **kwargs):
            pass

    requests_auth_stub.HTTPBasicAuth = _HTTPBasicAuth

    retry_stub = types.ModuleType("requests.packages.urllib3.util.retry")

    class _Retry:
        def __init__(self, *args, **kwargs):
            pass

    retry_stub.Retry = _Retry

    requests_packages_stub = types.ModuleType("requests.packages")
    urllib3_stub = types.ModuleType("requests.packages.urllib3")
    urllib3_util_stub = types.ModuleType("requests.packages.urllib3.util")

    requests_stub.adapters = requests_adapters_stub
    requests_stub.auth = requests_auth_stub
    requests_stub.packages = requests_packages_stub

    requests_packages_stub.urllib3 = urllib3_stub
    urllib3_stub.util = urllib3_util_stub
    urllib3_util_stub.retry = retry_stub

    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    monkeypatch.setitem(sys.modules, "requests.adapters", requests_adapters_stub)
    monkeypatch.setitem(sys.modules, "requests.auth", requests_auth_stub)
    monkeypatch.setitem(sys.modules, "requests.packages", requests_packages_stub)
    monkeypatch.setitem(sys.modules, "requests.packages.urllib3", urllib3_stub)
    monkeypatch.setitem(sys.modules, "requests.packages.urllib3.util", urllib3_util_stub)
    monkeypatch.setitem(sys.modules, "requests.packages.urllib3.util.retry", retry_stub)

    openai_stub = types.ModuleType("openai")
    openai_stub.api_key = None
    openai_stub.base_url = None
    openai_stub.BadRequestError = type("BadRequestError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", openai_stub)

    websockets_stub = types.ModuleType("websockets")
    monkeypatch.setitem(sys.modules, "websockets", websockets_stub)

    config_stub = types.ModuleType("config")
    config_stub.embedding_function = None
    config_stub.reset_db_indices = False
    config_stub.MODELS_CONFIG_LIST = None
    config_stub.elastic_url_port = "http://127.0.0.1:9200"
    config_stub.elastic_user = None
    config_stub.elastic_password = None
    config_stub.unique_id = None
    config_stub.discord_webhook = None
    config_stub.vector_store_type = "chroma"
    config_stub.chroma_persist_dir = "human_llm_vector_chromadb"
    monkeypatch.setitem(sys.modules, "config", config_stub)

    db_utils = importlib.import_module("utils.db_utils")
    monkeypatch.setattr(db_utils.UnifiedVectorDBConfig, "common_vectordb_embedding_function", None, raising=False)
    monkeypatch.delenv("HUMANLLM_DISABLE_EMBEDDINGS", raising=False)


def test_env_disables_embeddings(monkeypatch):
    db_utils = importlib.import_module("utils.db_utils")
    monkeypatch.setenv("HUMANLLM_DISABLE_EMBEDDINGS", "1")

    cfg = db_utils.UnifiedVectorDBConfig(embedding_function="text-embedding-ada-002")

    assert isinstance(cfg.common_vectordb_embedding_function, db_utils.DummyEmbeddingNonZero)


@pytest.mark.parametrize("sentinel", [None, "disabled", "off", "0", False])
def test_sentinel_disables_embeddings(monkeypatch, sentinel):
    db_utils = importlib.import_module("utils.db_utils")

    cfg = db_utils.UnifiedVectorDBConfig(embedding_function=sentinel)

    assert isinstance(cfg.common_vectordb_embedding_function, db_utils.DummyEmbeddingNonZero)


def test_prepare_configs_cli_switch(monkeypatch):
    import sys
    import types

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    llm_utils_stub = types.ModuleType("utils.llm_utils")
    llm_utils_stub.smart_print = lambda *args, **kwargs: None
    llm_utils_stub.smart_input = lambda *args, **kwargs: ""
    llm_utils_stub.import_functions_from_directory = lambda *args, **kwargs: {}
    llm_utils_stub.extract_function_code = lambda *args, **kwargs: ""
    llm_utils_stub.parse_learn_question = lambda *args, **kwargs: {}
    llm_utils_stub.get_success_value_in_text = lambda *args, **kwargs: 0
    llm_utils_stub.get_highest_score_index = lambda *args, **kwargs: 0
    monkeypatch.setitem(sys.modules, "utils.llm_utils", llm_utils_stub)

    env_pkg = types.ModuleType("env")
    env_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "env", env_pkg)

    env_env_module = types.ModuleType("env.env")

    class _EnvironmentManager:
        pass

    env_env_module.EnvironmentManager = _EnvironmentManager
    monkeypatch.setitem(sys.modules, "env.env", env_env_module)

    env_ir_pkg = types.ModuleType("env.IR_CPS_TechSynthesis")
    env_ir_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "env.IR_CPS_TechSynthesis", env_ir_pkg)
    monkeypatch.setitem(sys.modules, "env.IR_CPS_TechSynthesis.env", types.ModuleType("env.IR_CPS_TechSynthesis.env"))

    env_swe_pkg = types.ModuleType("env.SWEBench")
    env_swe_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "env.SWEBench", env_swe_pkg)
    monkeypatch.setitem(sys.modules, "env.SWEBench.env", types.ModuleType("env.SWEBench.env"))

    constants_stub = types.ModuleType("utils.constants")
    constants_stub.ELASTIC_DATABASE = "elasticsearch"
    constants_stub.CHROMA_DATABASE = "chroma"
    monkeypatch.setitem(sys.modules, "utils.constants", constants_stub)

    human_llm_stub = types.ModuleType("utils.human_llm")
    human_llm_stub.HumanLLM = type("HumanLLM", (), {})
    monkeypatch.setitem(sys.modules, "utils.human_llm", human_llm_stub)

    human_llm_config_stub = types.ModuleType("utils.human_llm_config")
    human_llm_config_stub.HumanLLMConfig = DummyHumanLLMConfig
    monkeypatch.setitem(sys.modules, "utils.human_llm_config", human_llm_config_stub)

    agents_stub = types.ModuleType("utils.agents")
    agents_stub.TaskIdentificationAgent = type("TaskIdentificationAgent", (), {})
    agents_stub.CodingAgent = type("CodingAgent", (), {})
    agents_stub.ValidationAgent = type("ValidationAgent", (), {})
    agents_stub.CapitalizationAgent = type("CapitalizationAgent", (), {})
    agents_stub.PlannerAgent = type("PlannerAgent", (), {})
    monkeypatch.setitem(sys.modules, "utils.agents", agents_stub)

    learn = importlib.import_module("learn")

    args = SimpleNamespace(
        port=1234,
        secret=False,
        proxy=False,
        pickle_name=None,
        disable_embeddings=True,
    )

    config = learn.prepare_configs(args)

    assert config.common_vectordb_config.embedding_function == "disabled"
