# utils/db_utils.py

import socket
from datetime import datetime
from typing import Any, Optional, Dict, List, Any
from jinja2 import Template
import json
import uuid

 # Dependency to be removed later
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings  # or langchain.embeddings if that’s what you use
from langchain.schema import Document
#from langchain_community.vectorstores import Chroma

try:
    from langchain_chroma import Chroma
except ImportError:
    print("langchain_chroma not installed")
    Chroma = None

try:
    from elasticsearch import Elasticsearch
    from langchain_community.vectorstores import ElasticsearchStore
except ImportError:
    print("Elasticsearch not installed")
    Elasticsearch, ElasticsearchStore = None, None

from utils.constants import ELASTIC_DATABASE, CHROMA_DATABASE

import logging
import re
import os
import requests
import time
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from requests.packages.urllib3.util.retry import Retry  # type: ignore

#––– Utility to produce a “unique collection id” if none is given –––#
def _default_unique_collection_id() -> str:
    """
    Exactly the same fallback used previously in llm_utils.py:
    f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    """
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    return f"{hostname}_{timestamp}"

class DummyEmbeddingNonZero:
    """Constant, non-zero embeddings; keeps DB happy but removes semantic ranking."""
    def __init__(self, dim: int = 16): self.dim = dim
    def embed_query(self, text: str): return [1.0] * self.dim
    def embed_documents(self, texts): return [[1.0] * self.dim for _ in texts]

DISABLED_EMBEDDING_STRINGS = {"disabled", "none", "off", "0"}


def _embedding_is_disabled(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in DISABLED_EMBEDDING_STRINGS
    return False


class UnifiedVectorDBConfig:
    """Configuration for the vector database."""
    common_vectordb_embedding_function = None

    def __init__(
        self,
        embedding_function: Optional[Any] = None,
        collection_name: Optional[str] = "human_llm_logs",
        persist_directory: Optional[str] = "human_llm_vectordb",
        reset_indices: bool = False,
        unique_collection_id: Optional[str] = None
    ):
        """Initialize VectorDBConfig."""
        self._requested_embedding_function = embedding_function
        self.embedding_function = embedding_function or DummyEmbeddingNonZero(dim=1)
        self.collection_name = collection_name.lower()
        self.persist_directory = persist_directory
        self.reset_indices = reset_indices

        # If the caller gave a unique_collection_id, use it; otherwise call our fallback.
        self.unique_collection_id = unique_collection_id or _default_unique_collection_id()

        # Put a default “db_type” (will be overridden at runtime if needed):
        self.db_type: str = CHROMA_DATABASE

        # Build out an OpenAI or HuggingFace embedding function exactly as llm_utils did:
        self.openai_embedding_function_name = "text-embedding-ada-002"
        self.set_common_vectordb_embedding_function()

        self.es_config = ElasticSearchDB_Config() if self.db_type == ELASTIC_DATABASE else None
        
    def set_common_vectordb_embedding_function(self):
        """Set the embedding function for the vector database."""
        if self.__class__.common_vectordb_embedding_function is not None:
            return

        disable_env = str(os.getenv("HUMANLLM_DISABLE_EMBEDDINGS", "")).lower() in ("1", "true", "yes", "on")
        requested = getattr(self, "_requested_embedding_function", None)

        if disable_env or _embedding_is_disabled(requested):
            self.__class__.common_vectordb_embedding_function = DummyEmbeddingNonZero(dim=1)
            print("\033[91m[Embeddings] Disabled — using DummyEmbeddingNonZero.\033[0m")
            return

        if isinstance(self.embedding_function, str):
            if self.embedding_function in ["OpenAIEmbeddings", "text-embedding-ada-002"]:
                if not os.getenv("OPENAI_API_KEY"):
                    self.__class__.common_vectordb_embedding_function = DummyEmbeddingNonZero(dim=1)
                    return
                self.__class__.common_vectordb_embedding_function = OpenAIEmbeddings(
                    model=self.embedding_function,
                    deployment=self.openai_embedding_function_name
                )
            elif self.embedding_function == "HuggingFaceEmbeddings":
                self.__class__.common_vectordb_embedding_function = HuggingFaceEmbeddings(
                    model_name="intfloat/e5-base-v2",
                    encode_kwargs={"normalize_embeddings": True}
                )
            else:
                self.__class__.common_vectordb_embedding_function = HuggingFaceEmbeddings(
                    model_name=self.embedding_function,
                    encode_kwargs={"normalize_embeddings": True},
                    model_kwargs={"trust_remote_code": True}
                )
        else:
            self.__class__.common_vectordb_embedding_function = self.embedding_function

    def set_unique_collection_id(self, unique_id):
        self.unique_collection_id = unique_id


class ElasticSearchDB_Config:
    def __init__(self):
        try: import config as cfg
        except: cfg = None
        self.es_url: str = 'http://127.0.0.1:9200' if not hasattr(cfg, 'elastic_url_port') else cfg.elastic_url_port
        self.es_user: Optional[str] = None if not hasattr(cfg, 'elastic_user') else cfg.elastic_user
        self.es_password: Optional[str] = None if not hasattr(cfg, 'elastic_password') else cfg.elastic_password


class UnifiedVectorDB:
    """Unified interface for vector databases (Elasticsearch or Chroma)."""
    db_connection_check_done = False

    def __init__(self, config: Optional[UnifiedVectorDBConfig]=None, check_db:bool=False):
        """Initialize UnifiedVectorDB."""
        # Copy config reference
        self.config: UnifiedVectorDBConfig = config

        # If the caller changed db_type to ELASTIC_DATABASE after config was built, ensure es_config exists
        if self.config.db_type == ELASTIC_DATABASE and self.config.es_config is None:
            self.config.es_config = ElasticSearchDB_Config()

        def friendly_collectionname_string(s):
            # Constraint 1: Truncate or pad the string to ensure it's between 3-63 characters
            s = s[:63].ljust(3, 'a')
            # Constraint 2: Ensure it starts and ends with an alphanumeric character
            if not s[0].isalnum():
                s = 'a' + s[1:]
            if not s[-1].isalnum():
                s = s[:-1] + 'a'
            # Constraint 3: Replace invalid characters with underscores
            s = re.sub(r'[^a-zA-Z0-9_-]', '_', s)
            # Constraint 4: Replace two consecutive periods with underscores
            s = s.replace('..', '__')
            # Constraint 5: Ensure it's not a valid IPv4 address
            if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', s):
                s = 'a' + s[1:]
            return s[:63]
        self.logger = logging.getLogger(__name__)
        
        self.elastic_client: Optional[Any] = None
        self.db: Optional[Any] = None
        self._collection: Optional[Any] = None
        
        if check_db:
            self.check_db()
        self.get_unique_id()

        if self.config.unique_collection_id is not None:
            self.config.collection_name = f"{self.config.unique_collection_id}_{self.config.collection_name}".lower()
        self.config.collection_name = friendly_collectionname_string(self.config.collection_name)

        if self.config.db_type == CHROMA_DATABASE:
            if Chroma is None:
                raise ImportError("Chroma vector store selected but 'chromadb' or langchain community support is not installed.")
        elif self.config.db_type == ELASTIC_DATABASE:
            if Elasticsearch is None or ElasticsearchStore is None:
                raise ImportError("Elasticsearch vector store selected but 'elasticsearch' library or LangChain ES support is not installed.")
        else:
            raise ValueError(f"Unsupported DB type: {self.config.db_type}")

        if self.config.db_type == CHROMA_DATABASE:
            self.config.persist_directory = friendly_collectionname_string(self.config.persist_directory)
            self.db = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.config.common_vectordb_embedding_function,
                persist_directory=self.config.persist_directory
            )
            self._collection = self.db._collection
        elif self.config.db_type == ELASTIC_DATABASE:
            self.elastic_client = Elasticsearch(
                self.config.es_config.es_url,
                http_auth=(
                    self.config.es_config.es_user,
                    self.config.es_config.es_password
                ) if (self.config.es_config.es_user not in [False, "", None]) else None,
                verify_certs=True,
                ssl_show_warn=False
            )
            self.db = ElasticsearchStore(
                index_name=self.config.collection_name,
                embedding=self.config.common_vectordb_embedding_function,
                es_connection=self.elastic_client,
                distance_strategy="COSINE"
            )
            self._collection = self.db
            embedding_test = self.config.common_vectordb_embedding_function.embed_query("test")
            embedding_size = len(embedding_test)
            if self.config.reset_indices:
                self.db.client.indices.delete(
                    index=self.config.collection_name,
                    ignore=[400, 404]
                )
            self.db._create_index_if_not_exists(
                index_name=self.config.collection_name,
                dims_length=embedding_size
            )
        else:
            raise ValueError(f"Unsupported DB type: {self.config.db_type}")

    def get_unique_id(self):
        """Generate or retrieve a unique ID for the collection."""
        from utils.human_llm import HumanLLMConfig
        if HumanLLMConfig().common_vectordb_config.unique_collection_id is None:
            HumanLLMConfig().common_vectordb_config.unique_collection_id = os.environ.get(
                'unique_id',
                f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
            )
        self.config.unique_collection_id =  HumanLLMConfig().common_vectordb_config.unique_collection_id
        return self.config.unique_collection_id

    def check_db(self):
        """Check the database connection."""
        if self.__class__.db_connection_check_done:
            return
        if self.config.db_type == ELASTIC_DATABASE:
            session = requests.Session()
            retry = Retry(total=5, backoff_factor=1)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            auth = (
                HTTPBasicAuth(self.config.es_config.es_user, self.config.es_config.es_password)
                if self.config.es_config.es_user else None
            )
            try:
                response = session.get(self.config.es_config.es_url, auth=auth, timeout=5, verify=False)
                response.raise_for_status()
                self.logger.info(f"Elasticsearch response: {response.text}")
                self.__class__.db_connection_check_done = True
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error: {e}\nURL: {self.config.es_config.es_url}\nCheck Elasticsearch and credentials.")
                exit(1)
        elif self.config.db_type == CHROMA_DATABASE:
            self.logger.info("Chroma DB check is not yet implemented")
            self.db_connection_check_done = True
        else:
            raise ValueError(f"Unsupported DB type: {self.config.db_type}")

    def _add_texts(self, texts, ids=None, metadatas=None):
        """Add texts to the database."""
        try:
            if isinstance(metadatas, dict):
                metadatas = {k: v for k, v in metadatas.items() if v is not None}
            elif isinstance(metadatas, list):
                metadatas = [{k: v for k, v in (m or {}).items() if v is not None} for m in metadatas]
            else:
                metadatas = None
            if self.config.db_type == CHROMA_DATABASE or self.config.db_type == ELASTIC_DATABASE:
                if isinstance(texts, list):
                    for i in range(len(texts)):
                        if not isinstance(texts[i], str):
                            texts[i] = str(texts[i])
                if metadatas is None:
                    return self.db.add_texts(texts=texts, ids=ids)
                return self.db.add_texts(texts=texts, ids=ids, metadatas=metadatas)
            else:
                self.logger.error(f"Unsupported DB type: {self.config.db_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error adding texts to database: {str(e)} / Texts: {texts} / IDs: {ids} / Metadatas: {metadatas}")
            return None

    def delete(self, ids):
        """Delete entries from the database by IDs."""
        if self.config.db_type == CHROMA_DATABASE:
            return self.db.delete(ids=ids)
        elif self.config.db_type == ELASTIC_DATABASE:
            return self.db.delete(ids=ids)

    def _similarity_search_with_score(self, query, k=1):
        """Perform a similarity search with scores."""
        if self.config.db_type == CHROMA_DATABASE:
            return self.db.similarity_search_with_score(query, k=k)
        elif self.config.db_type == ELASTIC_DATABASE:
            return self.db.similarity_search_with_score(query, k=(k if k <= 50 else 50))

    def _query(self, query_text="", k=1, metadata_filter=None, metadata_filter_or=False,
              custom_filter_chrome=None, custom_filter_es=None, sort_order=None):
        """Query the database with filters and sorting."""
        if self.config.db_type == CHROMA_DATABASE:
            filter_chroma = None
            if metadata_filter and custom_filter_chrome is None:
                conditions = []
                for key, value in metadata_filter.items():
                    sign = '$eq' if (isinstance(value, str) or isinstance(value, bool)) else '$in'
                    if sign == '$in' and not isinstance(value, (list, tuple)):
                        value = [value]
                    conditions.append({key: {sign: value}})
                # If only one condition, use it directly; otherwise wrap in $and or $or.
                if len(conditions) == 1:
                    filter_chroma = conditions[0]
                else:
                    filter_chroma = {('$or' if metadata_filter_or else '$and'): conditions}
                if sort_order in ['asc', 'desc']:
                    self.logger.warning("WARNING: sort not implemented for Chroma DB; performing in-memory sort")
            # Query the database using the filter (if any)
            # Check db size if the underlying store supports it; otherwise skip
            try:
                size = self.db._collection.count()
                if size == 0:
                    self.logger.warning("WARNING: Chroma DB is empty, returning empty results")
                    return []
                elif size < k:
                    k = size
            except Exception:
                pass
            try:
                results = self.db.similarity_search(query_text, k=k, filter=filter_chroma)
            except Exception as e:
                self.logger.error(f"Error querying Chroma DB: {str(e)}")
                return []
            # If a sort order is provided, sort the results in memory.
            if sort_order in ['asc', 'desc']:
                # Each item may be either Document or (Document, score) - Extract .metadata['time'] from whichever form it is.
                results = sorted( results, key=lambda x: ( x[0].metadata.get('time', "") if isinstance(x, tuple) else x.metadata.get('time', "") ), reverse=(sort_order == 'desc'))
            return results
        elif self.config.db_type == ELASTIC_DATABASE:
            if metadata_filter and "_id" in metadata_filter and not metadata_filter_or:
                # metadata_filter["_id"] might be a single string or a list
                id_values = (metadata_filter["_id"] if isinstance(metadata_filter["_id"], list) else [metadata_filter["_id"]])
                # Build a Document for each requested ID, with that ID in metadata and .id
                results = [Document(page_content="", metadata={"_id": doc_id}, id=doc_id) for doc_id in id_values]
                return results
            if metadata_filter and custom_filter_es is None:
                custom_filter_es = []
                for key, value in metadata_filter.items():
                    if key == '_id':
                        if isinstance(value, list):
                            custom_filter_es.append({"ids": {"values": value}})
                        else:
                            custom_filter_es.append({"ids": {"values": [value]}})
                    elif isinstance(value, dict) and any(k in value for k in ['gte', 'lte', 'gt', 'lt']):
                        custom_filter_es.append({"range": {f"metadata.{key}": value}})
                    elif isinstance(value, list):
                        custom_filter_es.append({"terms": {f"metadata.{key}": value}})
                    else:
                        custom_filter_es.append({"match": {f"metadata.{key}": value}})
                if metadata_filter_or:
                    custom_filter_es = {"bool": {"should": custom_filter_es}}
            if sort_order in ['asc', 'desc']:
                def custom_query(query_body: dict, query: str):
                    return {"query": {"bool": {"must": custom_filter_es}},
                            "sort": [{"metadata.time": {"order": sort_order}}]}
                results = self.db.similarity_search(query_text, k=(k if k <= 50 else 50), custom_query=custom_query)
            else:
                results = self.db.similarity_search(query_text, k=(k if k <= 50 else 50), filter=custom_filter_es)
            # Minimal propagation of _id into Document.id
            for doc in (item[0] if isinstance(item, tuple) else item for item in results):
                if not doc.id and "_id" in doc.metadata: doc.id = doc.metadata["_id"]
            return results

    def count(self):
        """Count the number of entries in the database."""
        if self.config.db_type == CHROMA_DATABASE:
            return self.db._collection.count()
        elif self.config.db_type == ELASTIC_DATABASE:
            response = self.db.client.count(index=self.config.collection_name, body={"query": {"match_all": {}}})
            return response['count']

    def clear(self):
        """Clear the database."""
        if self.config.db_type == CHROMA_DATABASE:
            self.db._collection.clear()
        if self.config.db_type == ELASTIC_DATABASE:
            response = self.db.client.delete_by_query(index=self.config.collection_name, body={"query": {"match_all": {}}})
            self.logger.info(f"Deleted {response['deleted']} documents from index {self.config.collection_name}")
            time.sleep(2)

    # ─── New method: log_agent_data ────────────────────────────────────────────────
    def log_agent_data(
        self,
        agent_name: str,
        data_key: str,
        data_value: Any,
        function_name: Optional[str] = None,
        task_id: bool = False,
        before_after: Optional[str] = None,
        user_id: Optional[str] = None,
        step_id: Optional[int] = None,
        task_type: Optional[str] = None,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store agent‐specific data as a new document in the vector DB.  Serializes data_value to JSON,
        builds a metadata dict (including agent_name, data_key, and any provided tags), and then calls add_texts().
        """
        # (1) Serialize the payload
        if isinstance(data_value, dict):
            serialized_data = json.dumps(data_value)
        else:
            # wrap primitive or list under a key of data_key
            serialized_data = json.dumps({data_key: data_value})

        # (2) Optionally generate an task_id UUID
        task_id = str(uuid.uuid4()) if task_id else None

        # (3) Build metadata tags
        tags: Dict[str, Any] = metadata.copy() if isinstance(metadata, dict) else {}
        params = {"agent_name": agent_name, "data_key": data_key, "function_name":function_name, "task_id": task_id, "before_after": before_after, "user_id": user_id, "step_id": step_id, "task_type": task_type, "score": score}
        # filter out any that are None
        for key, val in params.items():
            if val is not None:
                tags[key] = val
        # Always tag with a timestamp
        tags["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # (4) Delegate to add_texts()
        # Note: Chroma / Elasticsearch both accept `metadatas=[{…}]`.
        self._add_texts(texts=[serialized_data], metadatas=[tags])


    # ─── New method: get_agent_data ────────────────────────────────────────────────
    def get_agent_data(
        self,
        agent_name: Optional[str] = None,
        data_key: Optional[str] = None,
        task_id: Optional[str] = None,
        function_name: Optional[str] = None,
        before_after: Optional[str] = None,
        user_id: Optional[str] = None,
        step_id: Optional[int] = None,
        task_type: Optional[str] = None,
        score: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        sort_order: Optional[str] = None,
        k: int = 5,
        start_index: int = 0,
        end_index: Optional[int] = None,
        query_text: str = "*",
        new_storage: bool = True
    ):
        """
        Retrieve agent‐specific data from the vector DB.  Builds a metadata dict from all provided parameters,
        runs a `query(...)` under the hood, then paginates and parses results (JSON‐decoding payloads).

        Returns:
            parsed_list: list of deserialized payloads (dict or primitive)  
            raw_list:   list of full `Document`‐like objects returned by the underlying vector store
        """
        # (1) Build metadata dict to filter by
        filters: Dict[str, Any] = {}
        # params = ["agent_name", "data_key", "function_name", "task_id", "before_after", "user_id", "step_id", "task_type", "score"]
        # filters.update({k: locals()[k] for k in params if (k in locals()) and (locals()[k] is not None)})
        params = {"agent_name": agent_name, "data_key": data_key, "function_name":function_name, "task_id": task_id, "before_after": before_after, "user_id": user_id, "step_id": step_id, "task_type": task_type, "score": score}
        # filter out any that are None
        for key, val in params.items():
            if val is not None:
                filters[key] = val

        # Merge in any explicit metadata_filter
        if metadata_filter:
            filters.update(metadata_filter)

        # (2) Determine how many to fetch
        max_k = end_index if (end_index is not None) else k

        # (3) Query the DB
        results = self._query( query_text=query_text, k=max_k, metadata_filter=filters, sort_order=sort_order)

        # (4) Apply pagination
        paginated = (results[start_index:end_index] if end_index is not None else results[start_index:])

        # (5) Parse/deserialize each Document.page_content
        parsed_list: List[Any] = []
        for item in paginated:
            try:
                # `item.page_content` may be a JSON string
                temp = json.loads(item.page_content)
            except Exception:
                # fallback: return the raw page_content
                temp = item.page_content
            if new_storage:
                parsed_list.append(temp)
            else:
                # If caller wants the “old-style” raw data_key value:
                if isinstance(temp, dict) and (data_key in temp): #if isinstance(temp, dict) and (data_key in temp) and (len(temp) == 1):
                    parsed_list.append(temp[data_key])
                else:
                    parsed_list.append(None)

        return parsed_list, results

    # ───────────────────────── Few‑shot helpers (moved from HumanLLMConfig) ──

    def populate_few_shot_tags(self, prompt: str) -> str:
        """
        Expand every `few_shots:{…}` tag found in *prompt* by calling
        :py:meth:`get_few_shot_examples`.  The tag (including its JSON payload)
        is replaced by the generated examples.
        """
        if not prompt:
            return ""

        pattern = r"few_shots:\s*\{"
        for m in reversed(list(re.finditer(pattern, prompt, re.DOTALL))):
            # Find matching closing brace (manual balance → works for nested {})
            depth, i = 1, m.end()
            while depth and i < len(prompt):
                if prompt[i] == "{":
                    depth += 1
                elif prompt[i] == "}":
                    depth -= 1
                i += 1
            try:
                criteria_json = json.loads("{" + prompt[m.end():i - 1] + "}")
            except json.JSONDecodeError:
                continue  # leave tag untouched if JSON is malformed
            replacement = self.get_few_shot_examples([criteria_json])
            prompt = prompt[:m.start()] + replacement + prompt[i:]
        return prompt

    # ---- public: get a formatted block of examples --------------------------------

    def get_few_shot_examples(self, criteria_list) -> str:
        if not criteria_list:
            return ""

        blocks = []
        for spec in self._normalize_criteria(criteria_list):
            src      = spec["sources"]
            num      = spec["num"]
            q_text   = spec.get("query_text", "*")
            m_filter = spec.get("metadata_filter", {})

            if src == "learnt":
                m_filter = {**m_filter, "data_key": "learnt_task"}
            elif src == "failed":
                m_filter = {**m_filter, "data_key": "failed_task"}

            docs = self._query(query_text=q_text,
                               k=num,
                               metadata_filter=m_filter,
                               sort_order=spec.get("sort_order"))
            examples = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
            blocks.append(self.format_examples(examples, spec, spec.get("separators")))
        return "\n".join(blocks)

    # ---- helpers -----------------------------------------------------------------

    def _normalize_criteria(self, lst):
        """Ensure required keys & defaults."""
        out = []
        for c in lst:
            out.append({
                "sources":  c.get("sources", "learnt"),
                "num":      c.get("num", 5),
                "format":   c.get("format", "Json"),
                "query_text":        c.get("query_text", "*"),
                "metadata_filter":   c.get("metadata_filter", {}),
                "sort_order":        c.get("sort_order"),
                "similarity_search": c.get("similarity_search", False),
                "template":          c.get("template"),
                "separators":        c.get("separators"),
            })
        return out

    def format_examples(self,
                        examples,
                        criteria,
                        separators=None) -> str:
        """Return examples in **Json**, **Markdown** or **Jinja2** format."""
        if not examples:
            return ""

        fmt       = criteria.get("format", "Json").lower()
        template  = criteria.get("template")
        sep = separators or {
            "global_prefix": "\n<<",
            "global_suffix": ">>\n",
            "item_prefix":   "\n|",
            "item_suffix":   "|"
        }

        rendered = []
        for ex in examples:
            try:
                data = json.loads(ex["content"])
            except Exception:
                data = {"text": ex["content"]}

            if fmt == "json":
                rendered.append(json.dumps(data, indent=2))
            elif fmt == "markdown":
                rendered.append(self._to_markdown(data))
            elif fmt == "jinja2" and template:
                rendered.append(Template(template).render(**data))
            else:
                rendered.append(ex["content"])

        body = "".join(f"{sep['item_prefix']}{t}{sep['item_suffix']}" for t in rendered)
        return f"{sep['global_prefix']}{body}{sep['global_suffix']}"

    @staticmethod
    def _to_markdown(obj):
        return "\n".join(f"**{k}**: {v}" for k, v in obj.items())

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
# Export just these names when somebody does `from db_utils import *`
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
__all__ = [
    "UnifiedVectorDBConfig",
    "UnifiedVectorDB",
    "ElasticSearchDB_Config",
]
