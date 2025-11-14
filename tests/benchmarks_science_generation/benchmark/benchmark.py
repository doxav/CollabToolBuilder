#!/usr/bin/env python3
"""
Unified benchmarking entry-point for

    • STORM            (stanford-oval/storm @45ee413)
    • Co-STORM         (storm CLI)
    • Co-STORM-Graph   (storm-swarm HEAD)
    • WriteHERE        (principia-ai/WriteHERE @e6a86f)

Datasets
    • FreshWiki
    • WildSeek
    • document_embedding_analysis (DEA)

Metrics
    • knowledge_storm.collaborative_storm.costorm_eval.evaluate_run
    • DEA: embedding-based rubric via VoyagerEnvIR_CPS_TechSynthesis

Outputs
    • One JSON summary per <dataset, method> in results/
    • One aggregate CSV  : results/all_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
import re
import builtins
import importlib

import pickle
import os.path as osp


from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ctb = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../CollabToolBuilder"))
if ctb not in sys.path: sys.path.append(ctb)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../document_embedding_analysis")
    )
)

from config import OPENAI_API_KEY , embedding_function
from tests.benchmarks_science_generation.benchmark.doc_eval import evaluate_document
from tests.benchmarks_science_generation.benchmark.parser_config import costorm_parser, storm_parser, ai_scientist_parser, agent_laboratory_parser


from hllm_utils.human_llm import HumanLLM
from hllm_utils.human_llm_config import HumanLLMConfig

from common.doc_wiki import DocWiki

from knowledge_storm.human_llm_adapter import HumanLLMAdapter
import knowledge_storm.lm as lm_module

MODEL = "gpt-5-nano"  # Default model for STORM and Co-STORM
# MODEL = "gpt-4o-mini-2024-07-18"
# ────────────────────────── CONSTANTS ──────────────────────────
FRESHWIKI_CSV_URL = (
    "https://huggingface.co/datasets/EchoShao8899/FreshWiki/resolve/main/topic_list.csv"
)
FRESHWIKI_JSON_BASE = (
    "https://huggingface.co/datasets/EchoShao8899/FreshWiki/resolve/main/json/"
)
WILDSEEK_JSON_URL = (
    "https://huggingface.co/datasets/YuchengJiang/WildSeek/resolve/main/data.json"
)

STORM_GIT = "https://github.com/stanford-oval/storm.git"
STORM_GIT_COMMIT = "45ee413"
SWARM_GIT = "https://github.com/doxav/storm-swarm.git"
WRITEHERE_GIT = "https://github.com/doxav/WriteHERE.git"
WRITEHERE_GIT_COMMIT = "HEAD"
AISCIENTIST_GIT = "https://github.com/SakanaAI/AI-Scientist-v2.git"
AISCIENTIST_GIT_COMMIT = "HEAD"
AGENTLAB_GIT = "https://github.com/SamuelSchmidgall/AgentLaboratory.git"
AGENTLAB_GIT_COMMIT = "123444b7690aedbe895ec68c0b69b6fe65d2a22a"
# ────────────────────────── GLOBAL PATHS ──────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
SOURCES_DIR = CURRENT_DIR / "ext"
SOURCES_DIR.mkdir(exist_ok=True)
RESULT_DIR = CURRENT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Paths for external repos (filled in main)
STORM_DIR: Path | None = None
SWARM_DIR: Path | None = None
WRITEHERE_DIR: Path | None = None
AISCIENTIST_DIR: Path | None = None
AGENTLABORATORY_DIR: Path | None = None

costorm_gpt = None


# Cache for DEA file paths
DEA_FILES: Dict[str, Path] = {}
FRESHWIKI_DEA_FILES = {}
# ───────────────────────── DEFAULT PARAMS ───────────────────────────────────────────
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Storm": {"retriever": "searxng", "storm_path": None},
    "CoStorm": {"retriever": "searxng", "total_conv_turn": 3, "storm_path": None},
    "CoStormGraph": {},
    "AIScientist": {
        "model_writeup": MODEL,
        "model_citation": MODEL,
        "model_review": MODEL,
        "model_agg_plots": MODEL,
        "num_cite_rounds": "20",
    },
    "AgentLaboratory": {
        
    },
    "AgentLaboratory_HumanLLM": {
        "human_llm_parameters": [
            {
                "agent_name": "llm-backend"
            },
            {
                "agent_name": "lite-review-backend"
            },
        ]
    },
    "WriteHere": {
        "writehere_model": "gpt-5-nano",
        "engine_backend": "SearXNG",
        "writehere_path": None,
    },
}

# ────────────────────────── THIRD-PARTY IMPORTS ──────────────────────────
try:
    from env.VoyagerEnvIR_CPS_TechSynthesis import VoyagerEnvIR_CPS_TechSynthesis
    from env.env import EnvironmentManager
except ImportError:
    VoyagerEnvIR_CPS_TechSynthesis = None  # type: ignore

from tests.benchmarks_science_generation.benchmark.doc_eval import evaluate_run
from knowledge_storm.lm import LitellmModel

try:
    import requests
except ImportError:  # Give a gentle hint rather than crashing
    requests = None


class StopCurrentIteration(Exception):
    pass

# ────────────────────────── HELPERS ──────────────────────────
def clone_repo(
    dest: Path, repo: str, *, commit: str | None = None, branch: str | None = None
) -> None:
    """Clone *repo* into *dest* (if absent) and checkout *commit* or *branch*."""
    if dest.exists() and (dest / ".git").exists():
        return
    logging.info("Cloning %s → %s …", repo, dest)
    subprocess.run(
        # ["git", "clone", "--depth", "1", repo, str(dest)],
        ["git", "clone", repo, str(dest)],
        check=True,
        stdout=subprocess.PIPE,
    )
    if commit:
        subprocess.run(
            ["git", "checkout", commit],
            cwd=str(dest),
            check=True,
            stdout=subprocess.PIPE,
        )
    elif branch:
        subprocess.run(
            ["git", "checkout", branch],
            cwd=str(dest),
            check=True,
            stdout=subprocess.PIPE,
        )


def _make_openai_lm(model: str, max_tokens: int = 2048) -> LitellmModel:
    """Return a LitellmModel (wrapper around OpenAI chat-completion)."""
    return LitellmModel(
        model=model,
        max_tokens=max_tokens,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=1.0,
        top_p=0.9,
    )


def _evaluate_and_dump(
    *,
    topic: str,
    report: str,
    turns: List[Dict[str, Any]],
    method: str,
    dataset: str,
    out_dir: Path,
    solution=None,
) -> Dict[str, Any]:
    """Run evaluation and write raw / eval JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{dataset}_{method}_{abs(hash(topic))}.json"
    raw_path.write_text(
        json.dumps({"report": report, "turns": turns}, ensure_ascii=False),
        encoding="utf-8",
    )

    # 1) Keep existing aggregated run evaluator (turn metrics, etc.)
    scores: Dict[str, Any] = evaluate_run(raw_path)

    # 2) Add proper, reference-aware document metrics when we have a gold (DEA)
    if solution:
        doc_scores = evaluate_document(report, turns, solution)
        scores["document"] = doc_scores
        return scores
    # DEA extra rubric
    if dataset == "DEA" and VoyagerEnvIR_CPS_TechSynthesis:
        gold_fp = DEA_FILES.get(topic)
        if gold_fp and gold_fp.exists():
            try:
                env = EnvironmentManager(
                    env_type="techsynthesis",
                    title=topic,
                    context="",
                    target_file_path=str(gold_fp),
                ).get_environment()
                env.reset()
                # The report is plain text → parse via GetFromText (falls back to LaTeX internally)
                try:
                    env.synthesis_manager.GetFromLatex(report)
                except Exception:
                    env.synthesis_manager.GetFromText(report)

                detailed = env.synthesis_manager.get_distance_to_targetJSON()
                global_scores = env.get_score()

                rubric: Dict[str, Any] = {}
                if isinstance(detailed, dict):
                    rubric.update(detailed)
                if isinstance(global_scores, dict):
                    rubric.update({f"global_{k}": v for k, v in global_scores.items()})
                scores["latex_rubric"] = rubric
            except Exception as exc:  # pragma: no cover
                logging.warning("DEA rubric failed for %s – %s", topic, exc)

    # (raw_path.with_suffix(".eval.json")).write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    eval_path = raw_path.with_name(raw_path.name + ".eval.json")
    eval_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def get_last_id(base_path: str) -> int:
    """
    Scans all subdirectories in `base_path`, finds numeric prefixes (e.g., '0000001_'),
    and returns the next numeric ID.
    """
    id_pattern = re.compile(
        r"^(\d+)_"
    )  # matches one or more digits followed by underscore

    max_id = 0

    for root, dirs, files in os.walk(base_path):
        for dirname in dirs:
            match = id_pattern.match(dirname)
            if match:
                num_id = int(match.group(1))
                max_id = max(max_id, num_id)

    return max_id

def evaluate_document_and_save_metrics(
    topic:str,
    report:str,
    turns:Any,
    folder:str | Path,
    file_name:str = "metrics",
    dataset:str | Path = "Wikipedia",
    custom_dataset_path:str | Path = None,
):
    dataset = str(dataset).lower().strip()
    # TODO: change solution retrieval to use get the real topic
    if dataset == "wikipedia":
        intent, solution = get_solution_from_wikipedia(topic)
    if dataset == "dea":
        DEA_FOLDER = custom_dataset_path or Path("benchmark/dea")
        intent, solution = get_solution_from_dea(topic, DEA_FOLDER)
    metrics = evaluate_document(report, turns,solution)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    json_file = folder / f"{file_name}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics

# TODO: move to a proprer place
def get_human_llm_config(human_llm: HumanLLM) -> Dict[str, Any]:
    """
    Extract the dynamic configuration of a HumanLLM instance.
    Args:
        human_llm (HumanLLM): The HumanLLM instance to extract configuration from.
    Returns:
        Dict[str, Any]: The dynamic configuration of the HumanLLM instance.
    """
    return {
        "agent_name": human_llm.agent_name,
        "automation": human_llm.automation,
        "dynamic_llm_config": human_llm.dynamic_llm_config,
        "default_model": human_llm.default_llm.model_name,
        "premium_model": human_llm.premium_llm.model_name,
    }
     
def saved_config_human_llm(list_human_llm: List[HumanLLM], output_path: str | Path):
    """
    Save the configuration of HumanLLM instances to a JSON file.
    Args:
        list_human_llm (List[HumanLLM]): List of HumanLLM instances.
        output_path (str | Path): Path to save the configuration JSON file.
    """
    config_data = [get_human_llm_config(human_llm) for human_llm in list_human_llm]
    output_path = Path(output_path) / "human_llm_config.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
        
def create_human_llm_from_config(config: Dict[str, Any]) -> HumanLLM:
    """
    Create a HumanLLM instance from a configuration dictionary.
    Args:
        config (Dict[str, Any]): Configuration dictionary for the HumanLLM instance.
    Returns:
        HumanLLM: The created HumanLLM instance.
    """
    return HumanLLM(
        agent_name=config["agent_name"],
        automation=config["automation"],
        dynamic_llm_config=config["dynamic_llm_config"],
        llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
    )
_human_llm_config = None
_human_llm_parameters: List[Dict] = None
def get_default_human_llm_config():
    """
    Get the default HumanLLM configuration.
    Returns:
        Dict[str, Any]: The default HumanLLM configuration.
    """
    global _human_llm_config
    if _human_llm_config is None:
        _human_llm_config = HumanLLMConfig()
        _human_llm_config.use_websocket = False
        _human_llm_config.common_vectordb_config.embedding_function = embedding_function
        _human_llm_config.initialize()
    return _human_llm_config


# ────────────────────────── DATASET LOADERS ──────────────────────────
def _ensure_requests():
    if requests is None:  # pragma: no cover
        raise RuntimeError(
            "`requests` is required – install with `pip install requests`"
        )

def _get_wikipedia_url(topic: str, logger) -> str:
    """
    Get the canonical Wikipedia URL for a topic using the wikipedia library.
    
    Args:
        topic: The topic to search for
        logger: Logger instance
        
    Returns:
        The Wikipedia URL for the topic
    """
    try:
        import wikipedia
        
        # Search for the topic to get the exact page title
        search_results = wikipedia.search(topic, results=1)
        if not search_results:
            raise wikipedia.exceptions.DisambiguationError("No search results", [])   
        page_title = search_results[0]
        
        # Get the Wikipedia page to ensure it exists and get the URL
        page = wikipedia.page(page_title)
        wiki_url = page.url
        
        logger.info(f"Found Wikipedia page: {page_title}")
        return wiki_url
        
    except wikipedia.exceptions.DisambiguationError as e:
        # If there are multiple options, use the first one
        logger.warning(f"Disambiguation for {topic}, using first option: {e.options[0]}")
        try:
            page = wikipedia.page(e.options[0])
            return page.url
        except Exception as e2:
            logger.error(f"Error with disambiguation option: {e2}")
            return f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            
    except wikipedia.exceptions.PageError:
        logger.warning(f"Wikipedia page not found for {topic}")
        return f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        
    except Exception as e:
        logger.warning(f"Error searching Wikipedia for {topic}: {e}")
        return f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"


def _get_dea_filename(topic: str, output_path: Path) -> Path:
    """
    Generate the DEA filename for a topic.
    
    Args:
        topic: The topic name
        output_path: The output directory path
        
    Returns:
        Path to the DEA file
    """
    safe_topic = "".join(
        c for c in topic if c.isalnum() or c in (" ", "-", "_")
    ).strip()
    safe_topic = safe_topic.replace("_", " ")
    return output_path / f"{safe_topic}.json"


def _load_existing_dea(topic: str, dea_filename: Path, logger) -> Tuple[str, Dict[str, Any]] | None:
    """
    Try to load an existing DEA file.
    
    Args:
        topic: The topic name
        dea_filename: Path to the DEA file
        logger: Logger instance
        
    Returns:
        Tuple of (intent, solution) if successful, None otherwise
    """
    if dea_filename.exists():
        logger.info(f"Loading existing DEA for: {topic}")
        try:
            solution = json.loads(dea_filename.read_text(encoding="utf-8"))
            FRESHWIKI_DEA_FILES[topic] = dea_filename
            intent = f'Write a wikipedia like article about "{topic}"'
            solution["target_file_path"] = os.path.abspath(str(dea_filename))
            return intent, solution
        except Exception as e:
            logger.error(f"Error loading existing DEA for {topic}: {e}")
    return None


def _create_dea_from_wikipedia(topic: str, wiki_url: str, dea_output_dir: Path | str, logger) -> Dict[str, Any] | None:
    """
    Create a DEA using DocWiki from a Wikipedia URL.
    
    Args:
        topic: The topic name
        wiki_url: The Wikipedia URL
        dea_output_dir: Directory to store DEA files
        logger: Logger instance
        
    Returns:
        DEA data if successful, None otherwise
    """
    try:
        doc = DocWiki(wiki_url, logger, output_dir=dea_output_dir)
        dea_data = doc.extract_plan_and_content(skip_if_exists=False)
        
        dea_filename = _get_dea_filename(topic, Path(dea_output_dir))
        FRESHWIKI_DEA_FILES[topic] = dea_filename
        logger.info(f"Successfully created DEA for: {topic}")
        
        dea_data["target_file_path"] = os.path.abspath(str(dea_filename))
        return dea_data
        
    except Exception as e:
        logger.error(f"Error creating DEA for {topic}: {e}")
        return None


def get_solution_from_wikipedia(
    topic: str,
    dea_output_dir: Path | str = "benchmark/dea/output/wikipedia",
) -> Tuple[str, Dict[str, Any]]:
    """
    Get a solution for a specific Wikipedia topic.
    
    This function:
    1. Checks if a DEA JSON exists for the topic
    2. If not, creates one using DocWiki with the Wikipedia URL
    3. Returns the intent and DEA structure for the topic
    
    Args:
        topic: The Wikipedia topic to get solution for
        dea_output_dir: Directory to store/load DEA files
        
    Returns:
        Tuple of (intent, solution_dict)
    """
    _ensure_requests()

    # Ensure output directory exists
    output_path = Path(dea_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Generate expected DEA filename
    dea_filename = _get_dea_filename(topic, output_path)

    # Check if DEA JSON already exists
    existing_solution = _load_existing_dea(topic, dea_filename, logger)
    if existing_solution:
        return existing_solution

    # If no DEA exists, create one using DocWiki
    logger.info(f"Creating DEA for: {topic}")

    # Get Wikipedia URL using the wikipedia library
    wiki_url = _get_wikipedia_url(topic, logger)

    # Create DEA using DocWiki
    dea_data = _create_dea_from_wikipedia(topic, wiki_url, dea_output_dir, logger)
    
    intent = f'Write a wikipedia like article about "{topic}"'
    
    if dea_data:
        return intent, dea_data
    else:
        # Final fallback - return empty solution
        logger.warning(f"Could not create or load solution for {topic}")
        return intent, {}


def load_freshwiki(
    dea_output_dir: Path | str = "benchmark/dea/output/wikipedia",
) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """
    The HuggingFace FreshWiki dataset is stored as:
        topic_list.csv  – master index
        json/*.json     – one file per article

    This function:
    1. Streams the CSV once
    2. For each article, checks if a DEA JSON exists
    3. If not, creates one using DocWiki with the Wikipedia URL
    4. Returns the DEA structure (similar to load_dea)
    """
    _ensure_requests()
    import csv as _csv

    # Ensure output directory exists
    output_path = Path(dea_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    resp = requests.get(FRESHWIKI_CSV_URL, timeout=15)
    resp.raise_for_status()

    for row in _csv.DictReader(resp.text.splitlines()):
        topic = row["topic"].strip()
        wiki_url = row.get("url", "").strip()

        # Generate expected DEA filename
        dea_filename = _get_dea_filename(topic, output_path)

        # Check if DEA JSON already exists
        existing_solution = _load_existing_dea(topic, dea_filename, logger)
        if existing_solution:
            intent, solution = existing_solution
            yield topic, intent, solution
            continue

        # If no DEA exists, create one using DocWiki
        logger.info(f"Creating DEA for: {topic}")

        # Use the provided URL or get one from Wikipedia library
        if not wiki_url:
            wiki_url = _get_wikipedia_url(topic, logger)

        # Create DEA using DocWiki
        dea_data = _create_dea_from_wikipedia(topic, wiki_url, dea_output_dir, logger)
        
        intent = f'Write a wikipedia like article about "{topic}"'
        
        if dea_data:
            yield topic, intent, dea_data
        else:
            # Fallback - try to get FreshWiki data
            json_url = f"{FRESHWIKI_JSON_BASE}{topic.replace(' ', '_')}.json"
            try:
                freshwiki_data = requests.get(json_url, timeout=10).json()
                yield topic, intent, freshwiki_data
            except Exception as e:
                logger.error(f"Error processing {topic}: {e}")
                yield topic, intent, {}

def load_wildseek() -> Iterable[Tuple[str, str, str]]:
    """
    WildSeek is a single JSON list with fields `topic`, `intent`, `domain`.
    """
    _ensure_requests()
    data = requests.get(WILDSEEK_JSON_URL, timeout=15).json()
    solution = None
    for rec in data:
        yield rec["topic"].strip(), rec.get("intent", ""), solution, None


def load_dea(dea_root: Path | str) -> Iterable[Tuple[str, str, str]]:
    """
    DEA JSONs live under <dea_root>/output/latex/*.json
    We cache file paths in DEA_FILES for evaluation.
    """
    root = Path(dea_root)
    data_dir = root / "output" / "latex"
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    for fp in data_dir.glob("*.json"):
        solution = json.loads(fp.read_text(encoding="utf-8"))
        title = solution["title"]
        DEA_FILES[title] = fp
        solution["target_file_path"] = os.path.abspath(str(fp))
        yield title, solution.get("context", ""), solution


def get_solution_from_dea(filename: str, dea_root: Path | str) -> dict:
    """
    Get a specific DEA solution by filename.
    
    Args:
        filename: Name of the JSON file (with or without .json extension)
        dea_root: Root directory of the DEA data
        
    Returns:
        dict: The solution data with added target_file_path
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    root = Path(dea_root)
    data_dir = root / "output" / "latex"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    # Add .json extension if not present
    if not filename.endswith('.json'):
        filename += '.json'
    
    file_path = data_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Solution file not found: {file_path}")
    
    solution = json.loads(file_path.read_text(encoding="utf-8"))
    solution["target_file_path"] = os.path.abspath(str(file_path))
    
    return solution.get("context", ""),solution



DATASET_LOADERS: Dict[str, Callable[..., Iterable[Tuple[str, str, any]]]] = {
    "FreshWiki": load_freshwiki,
    "WildSeek": load_wildseek,
    "DEA": load_dea,
}

# ────────────────────────── METHOD PATCH ──────────────────────────


def input_override(
    predefined_inputs: dict[str, str],
) -> Tuple[Callable[[str], str], Callable[[str], str]]:
    """
    Creates a pair of input functions to override and restore the built-in input behavior using predefined inputs.
    Args:
        predefined_inputs (Iterable[str]): An iterable of strings to be used as simulated user inputs.
    Returns:
        Tuple[Callable[[str], str], Callable[[str], str]]:
            - get_predefined_input: A function that returns the next value from predefined_inputs when called, or falls back to the original input if exhausted.
            - _original_input: The original built-in input function for restoration.
    Usage:
        Use get_predefined_input to simulate user input in testing scenarios, and _original_input to restore the original input behavior.
    """
    _original_input = builtins.input
    predefined_inputs = {k.strip(): v for k, v in predefined_inputs.items()}

    def get_predefined_input(input_str: str) -> str:
        """Simulate user input for the topic."""
        if input_str.strip() in predefined_inputs:
            return predefined_inputs[input_str.strip()]
        return _original_input(input_str)

    return get_predefined_input, _original_input


def SearXNG_patch():
    import knowledge_storm.rm as rm_module

    _original_SearXNG = rm_module.SearXNG

    class SearXNGPatched(_original_SearXNG):
        def __init__(self, *args, **kwargs):
            searxng_api_url = kwargs.pop("searxng_api_url", None) or os.environ.get(
                "SearXNG", "http://127.0.0.1:8080/search"
            )
            super().__init__(searxng_api_url=searxng_api_url, *args, **kwargs)

    rm_module.SearXNG = SearXNGPatched


def patch_default_model_storm_costorm(model_name: str = None):
    model_name = model_name or MODEL

    class OpenAiModelPatched(lm_module.OpenAIModel):
        def __init__(self, *args, **kwargs):
            kwargs["model"] = model_name
            super().__init__(*args, **kwargs)

    lm_module.OpenAIModel = OpenAiModelPatched

    class AzureOpenAiModelPatched(lm_module.AzureOpenAIModel):
        def __init__(self, *args, **kwargs):
            kwargs["model"] = model_name
            super().__init__(*args, **kwargs)

    lm_module.AzureOpenAIModel = AzureOpenAiModelPatched


def patch_lm_config_setter(lm_config_class, HUMAN_LLM_CONFIG):
    original_methods_dict = {}
    for method_name, human_llm in HUMAN_LLM_CONFIG.items():
        original_method = getattr(lm_config_class, method_name)
        original_methods_dict[method_name] = original_method

        def make_wrapper(original_method, human_llm):
            def wrapper(self, model):
                model.set_human_llm(human_llm)
                return original_method(self, model)

            return wrapper

        setattr(lm_config_class, method_name, make_wrapper(original_method, human_llm))
    return original_methods_dict


def revert_lm_config_setters(_original_methods, lm_config_class):
    for method_name, original_method in _original_methods.items():
        setattr(lm_config_class, method_name, original_method)
    _original_methods.clear()  # Clear storage after revert


def run_costorm_storm_with_human_llm(
    HUMAN_LLM_CONFIG, class_config, func, *args, **kwargs
):
    original_methods = patch_lm_config_setter(class_config, HUMAN_LLM_CONFIG)
    human_llm_list = [human_llm for _, human_llm in HUMAN_LLM_CONFIG.items()]
    
    _original_OpenAIModel = lm_module.OpenAIModel
    _original_AzureOpenAIModel = lm_module.AzureOpenAIModel

    lm_module.OpenAIModel = HumanLLMAdapter
    lm_module.AzureOpenAIModel = HumanLLMAdapter
    try:
        
        outp_dir = kwargs.get("outp_dir")
        subdirs = [d for d in outp_dir.iterdir() if d.is_dir()]
        if len(subdirs) != 0:
            outp_dir = subdirs[0]
        saved_config_human_llm(human_llm_list,outp_dir)
        return func(*args, **kwargs)  # Run the function with patched methods
    finally:
        lm_module.OpenAIModel = _original_OpenAIModel  # Restore original OpenAIModel
        lm_module.AzureOpenAIModel = (
            _original_AzureOpenAIModel  # Restore original AzureOpenAIModel
        )
        revert_lm_config_setters(original_methods, class_config)


def storm_patch(callback: Callable = None,custom_llm_list: list = [], max_steps: int = -1):
    """
    Patch STORM to add a turn to the conversation history.
    This is useful for STORM's report_writing function.
    """
    from knowledge_storm.storm_wiki.engine import STORMWikiRunner
    from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
    
    original_method = STORMWikiRunner.run
    
    
    class MyHandler(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self.step = 0
                    
        def on_dialogue_turn_end(self, dlg_turn, **kwargs):
            print(dlg_turn)
            if callback:
                metrics = None
                article_change = False
                callback(custom_llm_list,dlg_turn, self.step,article_change,metrics)
            if max_steps != -1 and self.step >= max_steps:
                raise StopCurrentIteration(f"Maximum steps reached, stopping iteration. step = {self.step}")
            self.step += 1

    def patched_add_turn(*args, **kwargs):
        return original_method(*args,callback_handler=MyHandler(),**kwargs)

    STORMWikiRunner.run = patched_add_turn
    return original_method

def revert_storm_patch(original_method):
    """
    Revert the patch applied by `storm_patch`.
    This is useful to restore the original behavior of STORM.
    """
    from knowledge_storm.storm_wiki.engine import STORMWikiRunner

    STORMWikiRunner.run = original_method


def costorm_patch(
    output_dir,
    topic,
    CoStormRunner,
    callback=None,
    custom_llm_list=None,
    max_steps: int = -1,
    intermediate_evaluate: bool = True,
    generate_every_n: int = 2,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
):
    """
    Patch Co-STORM to add a turn to the conversation history.
    This is useful for Co-STORM's report_writing function.
    """

    original_method = CoStormRunner.step
    obj = {"step": 0}  # Use a mutable object to keep track of the step
    def patched_add_turn(self, *args, **kwargs):

        article_change = intermediate_evaluate and obj["step"] > 0 and (obj["step"] % generate_every_n == 0)
        article = None
        metrics = None
        if article_change:
            article = self.generate_report()
            with open(output_dir / f"article_step_{obj['step']}.md", "w", encoding="utf-8") as f:
                f.write(article)
            metrics = evaluate_document_and_save_metrics(
                topic=topic,
                report=article,
                turns=None,
                folder=output_dir,
                file_name=f"metrics_{obj['step']}",
                dataset=dataset_for_metrics,
                custom_dataset_path=custom_dataset_path,
            )
        if callback:
            callback(custom_llm_list, article, obj["step"], article_change, metrics)
        if max_steps != -1 and obj["step"] >= max_steps:
            raise StopCurrentIteration(f"Maximum steps reached, stopping iteration. step = {obj}")
        obj["step"] += 1
        res = original_method(self, *args, **kwargs)
        if res is None:
            raise ValueError(f"CoStormRunner.step returned None, which is unexpected. step = {obj['step']}")
        return res

    CoStormRunner.step = patched_add_turn
    return original_method

def revert_costorm_patch(CoStormRunner,original_method):
    """
    Revert the patch applied by `costorm_patch`.
    This is useful to restore the original behavior of Co-STORM.
    """

    CoStormRunner.step = original_method

def writehere_patch_add_turn(
    output_dir,
    topic,
    GraphRunEngine,
    callback=None,
    custom_llm=None,
    max_steps: int = -1,
    intermediate_evaluate: bool = True,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
):
    """
    Patch WriteHERE to add a turn to the conversation history.
    This is useful for WriteHERE's report_writing function.
    """

    original_method = GraphRunEngine.forward_one_step_not_parallel
    turns = []
    unknown_step_count = {"step": 0}

    def patched_add_turn(self, *args, **kwargs):
        step = None
        log_fn = kwargs.get("log_fn")
        if isinstance(log_fn, str):
            split = log_fn.split("/")
            if len(split) > 0 and split[-1].isdigit():
                step = int(split[-1])
        if step is None:
            step = "UnknownStep_" + str(unknown_step_count["step"])
            unknown_step_count["step"] += 1
            
        article = self.memory.article
        article_change = article != "" and len(turns) == 0 or len(turns) > 0 and article != turns[-1]
        metrics = None
        if article_change:
            turns.append(article)
            if intermediate_evaluate:
                with open(output_dir / f"article_step_{step}.md", "w", encoding="utf-8") as f:
                    f.write(article)
                metrics = evaluate_document_and_save_metrics(
                    topic=topic,
                    report=article,
                    turns=[],
                    folder=output_dir,
                    file_name=f"metrics_{step}",
                    dataset=dataset_for_metrics,
                    custom_dataset_path=custom_dataset_path,
                )
        if callback:
            callback([custom_llm],article, step, article_change,metrics)
        if max_steps != -1 and step >= max_steps:
            raise StopCurrentIteration(f"Maximum steps reached, stopping iteration. step = {step}")
        return original_method(self, *args, **kwargs)

    GraphRunEngine.forward_one_step_not_parallel = patched_add_turn
    return original_method, turns


def writehere_revert_patch(GraphRunEngine,original_method):
    """
    Revert the patch applied by `writehere_patch_add_turn`.
    This is useful to restore the original behavior of WriteHERE.
    """

    setattr(GraphRunEngine, "forward_one_step_not_parallel", original_method)


# APPLY SOME GLOBAL PATCHES
SearXNG_patch()
patch_default_model_storm_costorm()


# ────────────────────────── METHOD RUNNERS ──────────────────────────
def run_storm(
    topic: str,
    *,
    prompt: str = "Write a comprehensive, 10000 words 2025 about %s",
    retriever: str = "searxng",
    storm_path: str = None,
    search_top_k: int = 5,
    retrieve_top_k: int = 3,
    max_conv_turn: int = 3,
    output_dir: str | Path = None,
    callback: Callable = None,
    max_steps: int = -1,
    custom_llm_list: list=[],
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
    **kwargs,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute STORM example script (`run_storm_wiki_gpt.py`)."""
    if prompt.count("%s") != 1:
        raise ValueError("Prompt must contain exactly one '%s' placeholder.")
    
    formatted_prompt = prompt % topic
    
    global STORM_DIR, STORM_GIT, STORM_GIT_COMMIT
    STORM_DIR = (storm_path and Path(storm_path)) or STORM_DIR or SOURCES_DIR / "storm"
    # check if STORM_DIR is a folder and is not empty, else clone it
    if STORM_DIR is None or not STORM_DIR.exists() or not any(STORM_DIR.iterdir()):
        clone_repo(STORM_DIR, STORM_GIT, commit=STORM_GIT_COMMIT)
    if not STORM_DIR or not STORM_DIR.exists():
        raise RuntimeError(
            "STORM_DIR is not set or does not exist. Please provide a valid path with --storm-path."
        )
    if str(STORM_DIR) not in sys.path:
        sys.path.insert(0, str(STORM_DIR))
    # secret.toml does not exist at the root of STORM_DIR, we create it with OPENAI_API_KEY="{OPENAI_API_KEY}"\nOPENAI_API_TYPE="openai"
    secret_file = STORM_DIR / "secrets.toml"
    if not secret_file.exists():
        secret_file.write_text(
            f'OPENAI_API_KEY="{OPENAI_API_KEY}"\nOPENAI_API_TYPE="openai"\n',
            encoding="utf-8",
        )

    # outp_dir = RESULT_DIR / "tmp_storm"
    # subdirs = [d for d in outp_dir.iterdir() if d.is_dir()]
    # outp_dir = subdirs[0]
    if not output_dir:
        xp_id = (
            SHARED_ID
            + "_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + topic.replace(" ", "_").replace("/", "_")[:50]
        )
        outp_dir = RESULT_DIR / "tmp_storm" / xp_id
        outp_dir.mkdir(parents=True, exist_ok=True)

    import examples.storm_examples.run_storm_wiki_gpt as run_storm_wiki_gpt

    args = storm_parser().parse_args(
        [
            "--output-dir",
            str(outp_dir),
            "--retriever",
            retriever,
            "--do-research",
            "--do-generate-outline",
            "--do-generate-article",
            "--do-polish-article",
            "--max-conv-turn",
            str(
                max_conv_turn
            ),  # default=3, "Maximum number of questions in conversational question asking.",
            "--search-top-k",
            str(
                search_top_k
            ),  # default=3, "Top k search results to consider for each search query.",
            "--retrieve-top-k",
            str(
                retrieve_top_k
            ),  # default=3, "Top k collected references for each section title.",
        ]
    )
    predefined_input = {"Topic: ": formatted_prompt, "Your utterance: ": ""}
    get_predefined_input, _original_input = input_override(
        predefined_input
    )  # feed Topic + <ENTER> (dummy user utterance)


    _original_method = storm_patch(callback=callback, custom_llm_list=custom_llm_list, max_steps=max_steps)
    builtins.input = get_predefined_input
    
    run_storm_wiki_gpt.main(args)
    
    revert_storm_patch(_original_method)
    builtins.input = _original_input  # restore original input function

    # find first subdirectory in outp_dir
    subdirs = [d for d in outp_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise RuntimeError(
            f"No output subdirectory found in {outp_dir}. Did the script run successfully?"
        )
    outp_dir = subdirs[0]  # take the first subdirectory
    report = (outp_dir / "storm_gen_article_polished.txt").read_text(encoding="utf-8")
    turns = json.loads(
        (outp_dir / "conversation_log.json").read_text(encoding="utf-8")
    )  # use conversation_log.json # llm_call_history.jsonl is malformed in engine.py loop to write f.write(json.dumps(call) + "\n") => not properly formed JSON

    metrics = evaluate_document_and_save_metrics(
        topic=topic,
        report=report,
        turns=turns,
        folder=outp_dir,
        dataset=dataset_for_metrics,
        custom_dataset_path=custom_dataset_path,
    )
    
    
    return report, turns, metrics


def run_storm_human_llm(
    topic: str,
    *,
    prompt: str = "Write a comprehensive, 10000 words 2025 about %s",
    retriever: str = "searxng",
    storm_path: str = None,
    search_top_k: int = 5,
    retrieve_top_k: int = 3,
    max_conv_turn: int = 3,
    conv_simulator_lm: HumanLLM = None,
    question_asker_lm: HumanLLM = None,
    outline_gen_lm: HumanLLM = None,
    article_gen_lm: HumanLLM = None,
    article_polish_lm: HumanLLM = None,
    human_llm_parameters: List[Dict] = None,
    callback: Callable = None,
    max_steps: int = -1,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
    **kwargs,
) -> Tuple[str, List[Dict[str, Any]]]:
    if prompt.count("%s") != 1:
        raise ValueError("Prompt must contain exactly one '%s' placeholder.")
    for human_llm_config in human_llm_parameters or []:
        human_llm = create_human_llm_from_config(human_llm_config)
        agent_name = human_llm_config["agent_name"]
        if "conv_simulator_lm" in agent_name:
            conv_simulator_lm = conv_simulator_lm or human_llm
        elif "question_asker_lm" in agent_name:
            question_asker_lm = question_asker_lm or human_llm
        elif "outline_gen_lm" in agent_name:
            outline_gen_lm = outline_gen_lm or human_llm
        elif "article_gen_lm" in agent_name:
            article_gen_lm = article_gen_lm or human_llm
        elif "article_polish_lm" in agent_name:
            article_polish_lm = article_polish_lm or human_llm
        else:
            raise ValueError(f"Unknown HumanLLM agent name for storm: {agent_name}")
    
    if not conv_simulator_lm:
        conv_simulator_lm = HumanLLM(
            agent_name="conv_simulator_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not question_asker_lm:
        question_asker_lm = HumanLLM(
            agent_name="question_asker_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not outline_gen_lm:
        outline_gen_lm = HumanLLM(
            agent_name="outline_gen_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not article_gen_lm:
        article_gen_lm = HumanLLM(
            agent_name="article_gen_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not article_polish_lm:
        article_polish_lm = HumanLLM(
            agent_name="article_polish_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )

    HUMAN_LLM_CONFIG = {
        "set_conv_simulator_lm": conv_simulator_lm,
        "set_question_asker_lm": question_asker_lm,
        "set_outline_gen_lm": outline_gen_lm,
        "set_article_gen_lm": article_gen_lm,
        "set_article_polish_lm": article_polish_lm,
    }
    custom_llm_list = [
        conv_simulator_lm,
        question_asker_lm,
        outline_gen_lm,
        article_gen_lm,
        article_polish_lm,
    ]
    from knowledge_storm import STORMWikiLMConfigs
    
    xp_id = (
            SHARED_ID
            + "_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + topic.replace(" ", "_").replace("/", "_")[:50]
        )
    outp_dir = RESULT_DIR / "tmp_storm" / xp_id
    outp_dir.mkdir(parents=True, exist_ok=True)
    
    return run_costorm_storm_with_human_llm(
        HUMAN_LLM_CONFIG=HUMAN_LLM_CONFIG,
        class_config=STORMWikiLMConfigs,
        func=run_storm,
        topic=topic,
        prompt=prompt,
        retriever=retriever,
        storm_path=storm_path,
        search_top_k=search_top_k,
        retrieve_top_k=retrieve_top_k,
        max_conv_turn=max_conv_turn,
        outp_dir=outp_dir,
        callback=callback,
        max_steps=max_steps,
        custom_llm_list=custom_llm_list,
        dataset_for_metrics=dataset_for_metrics,
        custom_dataset_path=custom_dataset_path,
    )


def run_costorm(
    topic: str,
    *,
    prompt: str = "Write a comprehensive, 10000 words 2025 about %s",
    retriever: str = "searxng",
    total_conv_turn: int = 5,
    outp_dir: str | Path = None,
    callback = None,
    max_steps: int = -1,
    custom_llm_list: list=[],
    intermediate_evaluate: bool = True,
    generate_every_n: int = 2,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
    **kwargs,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute Co-STORM example script with a single dummy user turn."""
    if prompt.count("%s") != 1:
        raise ValueError("Prompt must contain exactly one '%s' placeholder.")
    formatted_prompt = prompt % topic
    
    if not outp_dir:
        xp_id = (
            SHARED_ID
            + "_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + topic.replace(" ", "_").replace("/", "_")[:50]
        )
        outp_dir = RESULT_DIR / "tmp_costorm" / xp_id
        outp_dir.mkdir(parents=True, exist_ok=True)

    from knowledge_storm.collaborative_storm.engine import CoStormRunner

    _original_method = costorm_patch(outp_dir,topic, CoStormRunner,callback=callback,
                                     custom_llm_list=custom_llm_list, max_steps=max_steps,
                                     intermediate_evaluate=intermediate_evaluate,
                                     generate_every_n=generate_every_n,
                                     dataset_for_metrics=dataset_for_metrics,
                                     custom_dataset_path=custom_dataset_path)

    import examples.costorm_examples.run_costorm_gpt as costorm_gpt

    
    args = costorm_parser().parse_args(
        [
            "--output-dir",
            str(outp_dir),
            "--retriever",
            retriever,
            "--total_conv_turn",
            str(total_conv_turn),
        ]
    )
    predefined_input = {
        "Topic: ": formatted_prompt,
        "Your utterance: ": "",
    }  # predefined input for the topic

    get_predefined_input, _original_input = input_override(
        predefined_input
    )  # feed Topic + <ENTER> (dummy user utterance)

    builtins.input = get_predefined_input
    try:
        costorm_gpt.main(args)
    finally:
        revert_costorm_patch(CoStormRunner,_original_method)  # restore original method

    builtins.input = _original_input  # restore original input function
    report = (outp_dir / "report.md").read_text(encoding="utf-8")
    instance_dump = json.loads(
        (outp_dir / "instance_dump.json").read_text(encoding="utf-8")
    )
    turns = instance_dump.get("conversation_history", [])

    metrics = evaluate_document_and_save_metrics(
        topic=topic,
        report=report,
        turns=turns,
        folder=outp_dir,
        dataset=dataset_for_metrics,
        custom_dataset_path=custom_dataset_path,
    )
    
    return report, turns, metrics


def run_costorm_human_llm(
    topic: str,
    *,
    prompt: str = "Write a comprehensive, 10000 words 2025 about %s",
    retriever: str = "searxng",
    total_conv_turn: int = 5,
    storm_path: str = None,
    question_answering_lm: HumanLLM = None,
    discourse_manage_lm: HumanLLM = None,
    utterance_polishing_lm: HumanLLM = None,
    warmstart_outline_gen_lm: HumanLLM = None,
    question_asking_lm: HumanLLM = None,
    knowledge_base_lm: HumanLLM = None,
    human_llm_parameters: List[Dict] = None,
    callback = None,
    max_steps: int = -1,
    intermediate_evaluate: bool = True,
    generate_every_n: int = 2,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
    **kwargs,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute Co-STORM example script with an replace LLM with HumanLLM."""
    if prompt.count("%s") != 1:
        raise ValueError("Prompt must contain exactly one '%s' placeholder.")
    for human_llm_config in human_llm_parameters or []:
        human_llm = create_human_llm_from_config(human_llm_config)
        agent_name = human_llm_config["agent_name"]
        if "question_answering_lm" in agent_name:
            question_answering_lm = question_answering_lm or human_llm
        elif "discourse_manage_lm" in agent_name:
            discourse_manage_lm = discourse_manage_lm or human_llm
        elif "utterance_polishing_lm" in agent_name:
            utterance_polishing_lm = utterance_polishing_lm or human_llm
        elif "warmstart_outline_gen_lm" in agent_name:
            warmstart_outline_gen_lm = warmstart_outline_gen_lm or human_llm
        elif "question_asking_lm" in agent_name:
            question_asking_lm = question_asking_lm or human_llm
        elif "knowledge_base_lm" in agent_name:
            knowledge_base_lm = knowledge_base_lm or human_llm
        else:
            raise ValueError(f"Unknown HumanLLM agent name for costorm: {agent_name}")
    if not question_answering_lm:
        question_answering_lm = HumanLLM(
            agent_name="question_answering_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not discourse_manage_lm:
        discourse_manage_lm = HumanLLM(
            agent_name="discourse_manage_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not utterance_polishing_lm:
        utterance_polishing_lm = HumanLLM(
            agent_name="utterance_polishing_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not warmstart_outline_gen_lm:
        warmstart_outline_gen_lm = HumanLLM(
            agent_name="warmstart_outline_gen_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not question_asking_lm:
        question_asking_lm = HumanLLM(
            agent_name="question_asking_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )
    if not knowledge_base_lm:
        knowledge_base_lm = HumanLLM(
            agent_name="knowledge_base_lm",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )

    HUMAN_LLM_CONFIG = {
        "set_question_answering_lm": question_answering_lm,
        "set_discourse_manage_lm": discourse_manage_lm,
        "set_utterance_polishing_lm": utterance_polishing_lm,
        "set_warmstart_outline_gen_lm": warmstart_outline_gen_lm,
        "set_question_asking_lm": question_asking_lm,
        "set_knowledge_base_lm": knowledge_base_lm,
    }

    from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs
    
    xp_id = (
            SHARED_ID
            + "_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + topic.replace(" ", "_").replace("/", "_")[:50]
        )
    outp_dir = RESULT_DIR / "tmp_costorm" / xp_id
    outp_dir.mkdir(parents=True, exist_ok=True)
    llm_list = [
        question_answering_lm,
        discourse_manage_lm,
        utterance_polishing_lm,
        warmstart_outline_gen_lm,
        question_asking_lm,
        knowledge_base_lm,
    ]
    return run_costorm_storm_with_human_llm(
        HUMAN_LLM_CONFIG=HUMAN_LLM_CONFIG,
        class_config=CollaborativeStormLMConfigs,
        func=run_costorm,
        topic=topic,
        prompt=prompt,
        retriever=retriever,
        total_conv_turn=total_conv_turn,
        storm_path=storm_path,
        outp_dir=outp_dir,
        callback=callback,
        custom_llm_list=llm_list,
        max_steps=max_steps,
        intermediate_evaluate  = intermediate_evaluate,
        generate_every_n = generate_every_n,
        dataset_for_metrics=dataset_for_metrics,
        custom_dataset_path=custom_dataset_path,
    )


def run_costorm_graph(
    topic: str,
    costorm_graph_path: str = None,
    *,
    total_conv_turn: int = 6,
    **kwargs
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run Co-STORM-Graph via Python API (storm-swarm override)."""
    global SWARM_DIR, SOURCES_DIR, SWARM_GIT
    SWARM_DIR = (
        (costorm_graph_path and Path(costorm_graph_path))
        or SWARM_DIR
        or SOURCES_DIR / "storm-swarm"
    )
    if not SWARM_DIR.exists() or not any(SWARM_DIR.iterdir()):
        clone_repo(SWARM_DIR, SWARM_GIT)
        sys.path.insert(0, str(SWARM_DIR))
    if not SWARM_DIR or not SWARM_DIR.exists():
        raise RuntimeError(
            "SWARM_DIR is not set or does not exist. Please provide a valid path with --costorm-graph-path."
        )
    if str(SWARM_DIR) not in sys.path:
        sys.path.insert(0, str(SWARM_DIR))
    secret_file = SWARM_DIR / "secrets.toml"
    if not secret_file.exists():
        secret_file.write_text(
            f'OPENAI_API_KEY="{OPENAI_API_KEY}"\nOPENAI_API_TYPE="openai"\n',
            encoding="utf-8",
        )

    from knowledge_storm.collaborative_storm.engine import (
        CollaborativeStormLMConfigs,
        RunnerArgument,
    )
    from knowledge_storm.collaborative_storm.graph_runner import CoStormGraphRunner
    from knowledge_storm.rm import BingSearch, DuckDuckGoSearchRM

    # Build a Co-STORM runner then orchestrate it with CoStormGraphRunner
    from knowledge_storm.collaborative_storm.engine import CoStormRunner, CollaborativeStormLMConfigs, RunnerArgument
    from knowledge_storm.logging_wrapper import LoggingWrapper
    from knowledge_storm.rm import SearXNG as DuckDuckGoSearchRM

    lm_cfg = CollaborativeStormLMConfigs()
    nano = _make_openai_lm("gpt-5-nano", max_tokens=8000)
    lm_cfg.set_question_answering_lm(nano)
    lm_cfg.set_discourse_manage_lm(nano)
    lm_cfg.set_utterance_polishing_lm(nano)
    lm_cfg.set_warmstart_outline_gen_lm(nano)
    lm_cfg.set_question_asking_lm(nano)
    lm_cfg.set_knowledge_base_lm(nano)

    args = RunnerArgument(topic=topic, retrieve_top_k=3, total_conv_turn=total_conv_turn)
    base = CoStormRunner(lm_config=lm_cfg, runner_argument=args, logging_wrapper=LoggingWrapper(lm_cfg), rm=DuckDuckGoSearchRM())
    g = CoStormGraphRunner(base)
    g.run(max_turns=total_conv_turn)
    report = base.generate_report()
    turns = [{"idx": i, "intent": t.utterance_type, "text": t.utterance} for i, t in enumerate(base.conversation_history)]
    return report, turns


def run_costorm_graph_human_llm(
    topic: str,
    *,
    prompt: str = "Write a comprehensive, 10000 words 2025 about %s",
    retriever: str = "searxng",
    total_conv_turn: int = 5,
    storm_path: str = None,
    question_answering_lm: HumanLLM = None,
    discourse_manage_lm: HumanLLM = None,
    utterance_polishing_lm: HumanLLM = None,
    warmstart_outline_gen_lm: HumanLLM = None,
    question_asking_lm: HumanLLM = None,
    knowledge_base_lm: HumanLLM = None,
    human_llm_parameters: List[Dict] = None,
    callback = None,
    max_steps: int = -1,
    intermediate_evaluate: bool = True,
    generate_every_n: int = 2,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
    **kwargs,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Co‑STORM‑Graph runner driven by HumanLLM adapters (trace optimizers inside HumanLLM).
    Honors max_steps as a turn bound and emits intermediate metrics like Co‑STORM.
    """
    if prompt.count("%s") != 1:
        raise ValueError("Prompt must contain exactly one '%s' placeholder.")

    # Map provided HumanLLMs to roles
    for human_llm_config in human_llm_parameters or []:
        human_llm = create_human_llm_from_config(human_llm_config)
        agent_name = human_llm_config["agent_name"]
        if "question_answering_lm" in agent_name:
            question_answering_lm = question_answering_lm or human_llm
        elif "discourse_manage_lm" in agent_name:
            discourse_manage_lm = discourse_manage_lm or human_llm
        elif "utterance_polishing_lm" in agent_name:
            utterance_polishing_lm = utterance_polishing_lm or human_llm
        elif "warmstart_outline_gen_lm" in agent_name:
            warmstart_outline_gen_lm = warmstart_outline_gen_lm or human_llm
        elif "question_asking_lm" in agent_name:
            question_asking_lm = question_asking_lm or human_llm
        elif "knowledge_base_lm" in agent_name:
            knowledge_base_lm = knowledge_base_lm or human_llm
        else:
            raise ValueError(f"Unknown HumanLLM agent name for costorm‑graph: {agent_name}")
    from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs, CoStormRunner, RunnerArgument
    from knowledge_storm.collaborative_storm.graph_runner import CoStormGraphRunner
    from knowledge_storm.logging_wrapper import LoggingWrapper
    from knowledge_storm.rm import SearXNG as DuckDuckGoSearchRM
    from knowledge_storm.human_llm_adapter import HumanLLMAdapter
    
    # Default HumanLLMs when not provided
    default_llm_list = get_default_human_llm_config().get_llmORchains_list()
    def _make_default(name):
        return HumanLLM(agent_name=name, automation=True, llmORchains_list=default_llm_list)
    question_answering_lm = question_answering_lm or _make_default("question_answering_lm")
    discourse_manage_lm   = discourse_manage_lm   or _make_default("discourse_manage_lm")
    utterance_polishing_lm= utterance_polishing_lm or _make_default("utterance_polishing_lm")
    warmstart_outline_gen_lm = warmstart_outline_gen_lm or _make_default("warmstart_outline_gen_lm")
    question_asking_lm    = question_asking_lm    or _make_default("question_asking_lm")
    knowledge_base_lm     = knowledge_base_lm     or _make_default("knowledge_base_lm")

    # LM config using HumanLLMAdapter for each role
    lm_cfg = CollaborativeStormLMConfigs()
    lm_cfg.set_question_answering_lm(HumanLLMAdapter(question_answering_lm))
    lm_cfg.set_discourse_manage_lm(HumanLLMAdapter(discourse_manage_lm))
    lm_cfg.set_utterance_polishing_lm(HumanLLMAdapter(utterance_polishing_lm))
    lm_cfg.set_warmstart_outline_gen_lm(HumanLLMAdapter(warmstart_outline_gen_lm))
    lm_cfg.set_question_asking_lm(HumanLLMAdapter(question_asking_lm))
    lm_cfg.set_knowledge_base_lm(HumanLLMAdapter(knowledge_base_lm))

    xp_id = (
        SHARED_ID + "_" + time.strftime("%Y%m%d-%H%M%S") + "_" + topic.replace(" ", "_").replace("/", "_")[:50]
    )
    outp_dir = RESULT_DIR / "tmp_costorm" / xp_id
    outp_dir.mkdir(parents=True, exist_ok=True)

    # Build runner and graph orchestrator
    args = RunnerArgument(topic=topic, retrieve_top_k=3, total_conv_turn=total_conv_turn)
    base = CoStormRunner(lm_config=lm_cfg, runner_argument=args, logging_wrapper=LoggingWrapper(lm_cfg), rm=DuckDuckGoSearchRM())
    g = CoStormGraphRunner(base)

    # Warm‑up + bounded turns
    g.run(max_turns=max_steps if max_steps and max_steps > 0 else total_conv_turn)

    # Intermediate evaluation (optional)
    if intermediate_evaluate and (max_steps or 0) > 0:
        # Evaluate final snapshot; for deeper per‑N steps, extend with per‑turn loop later.
        article = base.generate_report()
        with open(outp_dir / f"article_step_{(max_steps or 0)}.md", "w", encoding="utf-8") as f:
            f.write(article)
        evaluate_document_and_save_metrics(
            topic=topic,
            report=article,
            turns=None,
            folder=outp_dir,
            file_name=f"metrics_{(max_steps or 0)}",
            dataset=dataset_for_metrics,
            custom_dataset_path=custom_dataset_path,
        )
        if callback:
            callback([question_answering_lm, discourse_manage_lm, utterance_polishing_lm, warmstart_outline_gen_lm, question_asking_lm, knowledge_base_lm], article, (max_steps or 0), True, None)

    # Final artifacts
    report = base.generate_report()
    instance_dump = {
        "conversation_history": [t.to_dict() for t in getattr(base, "conversation_history", [])]
    }
    (outp_dir / "report.md").write_text(report, encoding="utf-8")
    (outp_dir / "instance_dump.json").write_text(json.dumps(instance_dump), encoding="utf-8")

    metrics = evaluate_document_and_save_metrics(
        topic=topic,
        report=report,
        turns=instance_dump.get("conversation_history", []),
        folder=outp_dir,
        dataset=dataset_for_metrics,
        custom_dataset_path=custom_dataset_path,
    )
    return report, instance_dump.get("conversation_history", []), metrics


def import_writehere(writehere_path: str = None) -> None:
    global WRITEHERE_DIR, SOURCES_DIR, WRITEHERE_GIT
    WRITEHERE_DIR = (
        (writehere_path and Path(writehere_path))
        or WRITEHERE_DIR
        or SOURCES_DIR / "WriteHERE"
    )
    if not WRITEHERE_DIR.exists() or not any(WRITEHERE_DIR.iterdir()):
        clone_repo(WRITEHERE_DIR, WRITEHERE_GIT, commit=WRITEHERE_GIT_COMMIT)
        sys.path.insert(0, str(WRITEHERE_DIR))
    if not WRITEHERE_DIR or not WRITEHERE_DIR.exists():
        raise RuntimeError(
            "WRITEHERE_DIR is not set or does not exist. Please provide a valid path with --writehere-path."
        )
    if str(WRITEHERE_DIR) not in sys.path:
        sys.path.insert(0, str(WRITEHERE_DIR))
    secret_file = WRITEHERE_DIR / "recursive/api_key.env"
    if not secret_file.exists():
        secret_file.write_text(
            f'OPENAI="{OPENAI_API_KEY}"\nSERPAPI=""\n', encoding="utf-8"
        )


def run_writehere(
    topic: str,
    intent: str,
    *,
    engine_backend: str,
    writehere_model: str = "gpt-5-nano",
    writehere_path: str = None,
    prompt: str = "Write a comprehensive, 10000 words 2025 about %s",
    tmp_dir: str| Path = None,
    callback: Callable = None,
    custom_llm = None,
    max_steps: int = -1,
    intermediate_evaluate: bool = True,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate a technical report via WriteHERE’s low-level `report_writing`."""
    # Accept either a templated prompt with one %s or a fully formatted prompt
    if prompt.count("%s") == 1:
        formated_prompt = prompt % topic
    else:
        formated_prompt = prompt
    import_writehere(writehere_path)

    from recursive.engine import report_writing

    assert WRITEHERE_DIR
    if not tmp_dir:
        tmp_dir = Path(
            os.path.abspath(
                os.path.join(
                    RESULT_DIR,
                    "tmp_writehere",
                    SHARED_ID
                    + "_"
                    + time.strftime("%Y%m%d-%H%M%S")
                    + "_"
                    + topic.replace(" ", "_").replace("/", "_")[:50],
                )
            )
        )
        tmp_dir.mkdir(exist_ok=True, parents=True)
    # generate a unique name based on timestamp data YMD-HMS
    inp = tmp_dir / "input.jsonl"
    outp = tmp_dir / "output.jsonl"
    turn_output = tmp_dir / "turns.json"

    goal = f"{topic}. {intent}" if intent else topic
    # Align searcher_type with selected engine for consistency
    searcher = "SearXNG" if engine_backend and engine_backend.lower() == "searxng" else "DuckDuckGoSearch"
    task = {
        "id": 1,
        "dependency": 0,
        "goal": f"{topic}. {intent}" if intent else topic,
        "task_type": "COMPOSITION",
        "length": 1,
        "model": MODEL or "gpt-5-nano",
        "prompt": formated_prompt,
        "searcher_type": searcher,
    }
    inp.write_text(json.dumps(task) + "\n", encoding="utf-8")
    from recursive.engine import GraphRunEngine

    original_writehere_func, turns = writehere_patch_add_turn(tmp_dir,topic,GraphRunEngine,
                                                              callback=callback,
                                                              custom_llm=custom_llm,max_steps=max_steps ,
                                                              intermediate_evaluate=intermediate_evaluate,
                                                              dataset_for_metrics=dataset_for_metrics,
                                                              custom_dataset_path=custom_dataset_path)
    report_writing(
        str(inp),
        str(outp),
        start=None,
        end=None,
        done_flag_file=None,
        global_use_model=writehere_model,
        engine_backend=engine_backend,
    )
    writehere_revert_patch(GraphRunEngine,original_writehere_func)

    res = outp.read_text(encoding="utf-8").splitlines()
    res = res[0]
    result = json.loads(res)
    article = result.get("article") or result.get("result") or ""

    with open(turn_output, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)

    metrics = evaluate_document_and_save_metrics(
        topic=topic,
        report=article,
        turns=turns,
        folder=tmp_dir,
        dataset=dataset_for_metrics,
        custom_dataset_path=custom_dataset_path,
    )

    return article, turns, metrics


def run_writehere_human_llm(
    topic: str,
    intent: str,
    *,
    engine_backend: str,
    writehere_model: str = "gpt-5-nano",
    writehere_path: str = None,
    prompt: str = "Write a comprehensive, 10000 words 2025 about %s",
    human_llm: HumanLLM = None,
    human_llm_parameters: Dict = None,
    callback: Callable = None,
    max_steps: int = -1,
    intermediate_evaluate: bool = True,
    dataset_for_metrics: str = "Wikipedia",
    custom_dataset_path: str | Path = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate a technical report via WriteHERE’s low-level `report_writing` using Human_llm."""
    # Accept prompt with or without a '%s' placeholder
    import_writehere(writehere_path)
    
    tmp_dir = Path(
        os.path.abspath(
            os.path.join(
                RESULT_DIR,
                "tmp_writehere",
                SHARED_ID
                + "_"
                + time.strftime("%Y%m%d-%H%M%S")
                + "_"
                + topic.replace(" ", "_").replace("/", "_")[:50],
            )
        )
    )
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    from recursive.llm.llm import OpenAIApiProxy

    if human_llm_parameters:
        human_llm = create_human_llm_from_config(human_llm_parameters)
    
    if not human_llm:
        human_llm = HumanLLM(
            agent_name="WriteHERE",
            automation=True,
            llmORchains_list=get_default_human_llm_config().get_llmORchains_list(),
        )

    human_llm_adapter = HumanLLMAdapter(human_llm=human_llm)

    _orig_call = OpenAIApiProxy.call

    def _patched_call(self, messages=None, **kwargs):
        
        # Recréer un prompt linéaire à partir des messages (simple concat)
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts)
        print(f"Prompt for HumanLLMAdapter:\n{prompt}\n")
        # Appel à HumanLLMAdapter (retourne une liste)
        response_texts = human_llm_adapter(prompt, **kwargs)
        content = response_texts[0] if response_texts else ""

        return [{"message": {"content": content}}]

    OpenAIApiProxy.call = _patched_call

    # Save the human_llm configuration to the tmp_dir
    saved_config_human_llm([human_llm],tmp_dir)
    
    res = run_writehere(
        topic,
        intent,
        engine_backend=engine_backend,
        writehere_model=writehere_model,
        writehere_path=writehere_path,
        prompt=prompt,
        tmp_dir=tmp_dir,
        callback=callback,
        custom_llm=human_llm,
        max_steps=max_steps,
        intermediate_evaluate=intermediate_evaluate,
        dataset_for_metrics=dataset_for_metrics,
        custom_dataset_path=custom_dataset_path,
    )
    OpenAIApiProxy.call = _orig_call  # Restore original call method

 

    return res



def run_ai_scientist(
    topic: str,
    intent: str,
    *,
    ai_scientist_path: str = None,
    output_dir: str = None,
    **kwargs
)-> Tuple[str, List[Dict[str, Any]]]:
    global AISCIENTIST_DIR, AISCIENTIST_GIT, AISCIENTIST_GIT_COMMIT
    AISCIENTIST_DIR = (ai_scientist_path and Path(ai_scientist_path)) or AISCIENTIST_DIR or SOURCES_DIR / "ai_scientist_v2"
    # check if AISCIENTIST_DIR is a folder and is not empty, else clone it
    if AISCIENTIST_DIR is None or not AISCIENTIST_DIR.exists() or not any(AISCIENTIST_DIR.iterdir()):
        clone_repo(AISCIENTIST_DIR, AISCIENTIST_GIT, commit=AISCIENTIST_GIT_COMMIT)
    if not AISCIENTIST_DIR or not AISCIENTIST_DIR.exists():
        raise RuntimeError(
            "AISCIENTIST_DIR is not set or does not exist. Please provide a valid path with --ai-scientist-path."
        )
    if str(AISCIENTIST_DIR) not in sys.path:
        sys.path.insert(0, str(AISCIENTIST_DIR))
    if not output_dir:
        xp_id = (
            SHARED_ID
            + "_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + topic.replace(" ", "_").replace("/", "_")[:50]
        )
        output_dir = RESULT_DIR / "tmp_ai_scientist" / xp_id
        output_dir.mkdir(parents=True, exist_ok=True)
    import ext.ai_scientist_v2.launch_scientist_bfts as run_ai_scientist

    idea = [{
        "Name": topic,
        "Title": intent,
    }]
    idea_path = str(AISCIENTIST_DIR) + "/ai_scientist/ideas/experiment.json"
    with open(idea_path, 'w') as f:
        json.dump(idea, f, indent=4)
    args_list = [
        "--load_ideas",
        idea_path,
    ]
    for model in ["model_writeup", "model_citation", "model_review", "model_agg_plots", "num_cite_rounds"]:
        if model in kwargs:
            args_list += [f"--{model}", kwargs[model]]

    args = ai_scientist_parser().parse_args(args_list)

    available_gpus = run_ai_scientist.get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")
    idea = ideas[args.idea_idx]
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"{output_dir}/experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)

    # Convert idea json to markdown file
    idea_path_md = osp.join(idea_dir, "idea.md")

    # If load_code is True, get the Python file with same name as JSON
    code = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
        else:
            print(f"Warning: Code file {code_path} not found")
    else:
        code_path = None

    run_ai_scientist.idea_to_markdown(ideas[args.idea_idx], idea_path_md, code_path)

    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            print(f"Warning: Dataset reference file {dataset_ref_path} not found")
            dataset_ref_code = None

    if dataset_ref_code is not None and code is not None:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code is not None and code is None:
        added_code = dataset_ref_code
    elif dataset_ref_code is None and code is not None:
        added_code = code
    else:
        added_code = None

    print(added_code)

    # Add code to idea json if it was loaded
    if added_code is not None:
        ideas[args.idea_idx]["Code"] = added_code

    # Store raw idea json
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    config_path = str(AISCIENTIST_DIR) + "/bfts_config.yaml"
    idea_config_path = run_ai_scientist.edit_bfts_config_file(
        config_path,
        idea_dir,
        idea_path_json,
    )

    # Load YAML
    import yaml
    with open(idea_config_path, 'r') as f:
        config = yaml.safe_load(f)
    config["report"]["model"] = MODEL;
    config["agent"]["code"]["model"] = MODEL
    config["agent"]["feedback"]["model"] = MODEL
    config["agent"]["vlm_feedback"]["model"] = MODEL
    # Save back
    with open(idea_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    

    run_ai_scientist.perform_experiments_bfts(idea_config_path)
    experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
    if os.path.exists(experiment_results_dir):
        shutil.copytree(
            experiment_results_dir,
            osp.join(idea_dir, "experiment_results"),
            dirs_exist_ok=True,
        )

    run_ai_scientist.aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)

    if os.path.exists(osp.join(idea_dir, "experiment_results")):
        shutil.rmtree(osp.join(idea_dir, "experiment_results"))
    
    run_ai_scientist.save_token_tracker(idea_dir)
    if not args.skip_writeup:
        writeup_success = False
        citations_text = run_ai_scientist.gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                writeup_success = run_ai_scientist.perform_writeup(
                    base_folder=idea_dir,
                    big_model=args.model_writeup,
                    page_limit=8,
                    citations_text=citations_text,
                )
            else:
                writeup_success = run_ai_scientist.perform_icbinb_writeup(
                    base_folder=idea_dir,
                    big_model=args.model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    run_ai_scientist.save_token_tracker(idea_dir)
    if not args.skip_review and not args.skip_writeup:
        # Perform paper review if the paper exists
        pdf_path = run_ai_scientist.find_pdf_path_for_review(idea_dir)
        if os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = run_ai_scientist.load_paper(pdf_path)
            client, client_model = run_ai_scientist.create_client(args.model_review)
            review_text = run_ai_scientist.perform_review(paper_content, client_model, client)
            review_img_cap_ref = run_ai_scientist.perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")
    
    print("Start cleaning up processes")
    # Kill all mp and torch processes associated with this experiment
    import psutil
    import signal

    # Get the current process and all its children
    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    # First try graceful termination
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Wait briefly for processes to terminate
    gone, alive = psutil.wait_procs(children, timeout=3)

    # If any processes remain, force kill them
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Additional cleanup: find any orphaned processes containing specific keywords
    keywords = ["python", "torch", "mp", "bfts", "experiment"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # Check both process name and command line arguments
            cmdline = " ".join(proc.cmdline()).lower()
            if any(keyword in cmdline for keyword in keywords):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
                if proc.is_running():
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    return None, None

def run_agent_laboratory(
    topic: str,
    *,
    agent_laboratory_path: str = None,
    llm_backend_in: str = None,
    lite_review_backend: str = None,
    output_dir: str = None,
    only_literature_review: bool = False,
    num_papers_to_write_override: int | None = None,
    num_papers_lit_review_override: int | None = None,
    **kwargs
)-> Tuple[str, List[Dict[str, Any]]]:
    global AGENTLABORATORY_DIR, AGENTLAB_GIT, AGENTLAB_GIT_COMMIT
    AGENTLABORATORY_DIR = (agent_laboratory_path and Path(agent_laboratory_path)) or AGENTLABORATORY_DIR or SOURCES_DIR / "agent_laboratory"
    # check if AISCIENTIST_DIR is a folder and is not empty, else clone it
    if AGENTLABORATORY_DIR is None or not AGENTLABORATORY_DIR.exists() or not any(AGENTLABORATORY_DIR.iterdir()):
        clone_repo(AGENTLABORATORY_DIR, AGENTLAB_GIT, commit=AGENTLAB_GIT_COMMIT)
    if not AGENTLABORATORY_DIR or not AGENTLABORATORY_DIR.exists():
        raise RuntimeError(
            "AGENTLABORATORY_DIR is not set or does not exist. Please provide a valid path with --ai-scientist-path."
        )
    if str(AGENTLABORATORY_DIR) not in sys.path:
        sys.path.insert(0, str(AGENTLABORATORY_DIR))
    if not output_dir:
        xp_id = (
            SHARED_ID
            + "_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + topic.replace(" ", "_").replace("/", "_")[:50]
        )
        output_dir = "results/tmp_agent_laboratory/" + str(xp_id)
        os.makedirs(output_dir, exist_ok=True)
    
    import yaml
    yaml_file = AGENTLABORATORY_DIR / "experiment_configs/MATH_agentlab.yaml"

    # 1. Load the YAML file
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    _default_model = "gpt-4o-mini"
    config['research-topic'] = topic
    config['api-key'] = OPENAI_API_KEY
    config['llm-backend'] = llm_backend_in or _default_model
    config['lit-review-backend'] = lite_review_backend or _default_model
    # Optional: bound runtime via overrides
    if num_papers_to_write_override is not None:
        config['num_papers_to_write'] = int(num_papers_to_write_override)
    if num_papers_lit_review_override is not None:
        config['num_papers_lit_review'] = int(num_papers_lit_review_override)

    # 3. Save it back to file
    with open(yaml_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    import benchmark.ext.agent_laboratory.utils as lab_utils
    import benchmark.ext.agent_laboratory.inference as lab_inference
    import benchmark.ext.agent_laboratory.agents as lab_agents
    import benchmark.ext.agent_laboratory.ai_lab_repo as run_agent_laboratory

    run_agent_laboratory.DEFAULT_LLM_BACKBONE = _default_model
    for name in dir(lab_utils):
        if not name.startswith("_") and callable(getattr(lab_utils, name)):
            setattr(lab_agents, name, getattr(lab_utils, name))
            setattr(run_agent_laboratory, name, getattr(lab_utils, name))

    args = run_agent_laboratory.parse_yaml(yaml_file)

    # Fix arXiv API issue: Patch extract_prompt to clean arXiv IDs
    # The arXiv API rejects IDs with "arXiv:" prefix, requiring clean format like "2412.05694"
    if not hasattr(lab_utils, '_extract_prompt_patched'):
        original_extract_prompt = lab_utils.extract_prompt
        
        def clean_extract_prompt(text, word):
            """Patched version that cleans arXiv IDs in FULL_TEXT commands"""
            result = original_extract_prompt(text, word)
            if word == "FULL_TEXT" and result and isinstance(result, str) and result.startswith('arXiv:'):
                result = result[6:]  # Remove 'arXiv:' prefix
            return result
        
        # Apply the patch to both utils and agents modules (agents uses a global name)
        lab_utils.extract_prompt = clean_extract_prompt
        lab_agents.extract_prompt = clean_extract_prompt
        lab_utils._extract_prompt_patched = True

    args = run_agent_laboratory.parse_yaml(yaml_file)
    
    llm_backend = args.llm_backend
    human_mode =  args.copilot_mode.lower() == "true" if type(args.copilot_mode) == str else args.copilot_mode
    compile_pdf = args.compile_latex.lower() == "true" if type(args.compile_latex) == str else args.compile_latex
    load_previous = args.load_previous.lower() == "true" if type(args.load_previous) == str else args.load_previous
    parallel_labs = args.parallel_labs.lower() == "true" if type(args.parallel_labs) == str else args.parallel_labs
    except_if_fail = args.except_if_fail.lower() == "true" if type(args.except_if_fail) == str else args.except_if_fail
    agentRxiv = args.agentRxiv.lower() == "true" if type(args.agentRxiv) == str else args.agentRxiv
    construct_agentRxiv = args.construct_agentRxiv.lower() == "true" if type(args.construct_agentRxiv) == str else args.construct_agentRxiv
    lab_index = int(args.lab_index) if type(args.construct_agentRxiv) == str else args.lab_index

    try: num_papers_to_write = int(args.num_papers_to_write.lower()) if type(args.num_papers_to_write) == str else args.num_papers_to_write
    except Exception: raise Exception("args.num_papers_lit_review must be a valid integer!")
    try: num_papers_lit_review = int(args.num_papers_lit_review.lower()) if type(args.num_papers_lit_review) == str else args.num_papers_lit_review
    except Exception: raise Exception("args.num_papers_lit_review must be a valid integer!")
    try: papersolver_max_steps = int(args.papersolver_max_steps.lower()) if type(args.papersolver_max_steps) == str else args.papersolver_max_steps
    except Exception: raise Exception("args.papersolver_max_steps must be a valid integer!")
    try: mlesolver_max_steps = int(args.mlesolver_max_steps.lower()) if type(args.mlesolver_max_steps) == str else args.mlesolver_max_steps
    except Exception: raise Exception("args.mlesolver_max_steps must be a valid integer!")
    if parallel_labs:
        num_parallel_labs = int(args.num_parallel_labs)
        print("="*20 , f"RUNNING {num_parallel_labs} LABS IN PARALLEL", "="*20)
    else: num_parallel_labs = 0

    api_key = (os.getenv('OPENAI_API_KEY') or args.api_key) if (hasattr(args, 'api_key') or os.getenv('OPENAI_API_KEY')) else None
    deepseek_api_key = (os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key) if (hasattr(args, 'deepseek_api_key') or os.getenv('DEEPSEEK_API_KEY')) else None
    if api_key is not None and os.getenv('OPENAI_API_KEY') is None: os.environ["OPENAI_API_KEY"] = args.api_key
    if deepseek_api_key is not None and os.getenv('DEEPSEEK_API_KEY') is None: os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key: raise ValueError("API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable.")

    if human_mode or args.research_topic is None: research_topic = input("Please name an experiment idea for AgentLaboratory to perform: ")
    else: research_topic = args.research_topic

    task_notes_LLM = list()
    task_notes = args.task_notes
    for _task in task_notes:
        for _note in task_notes[_task]:
            task_notes_LLM.append({"phases": [_task.replace("-", " ")], "note": _note})

    if args.language != "English":
        task_notes_LLM.append(
            {"phases": ["literature review", "plan formulation", "data preparation", "running experiments", "results interpretation", "report writing", "report refinement"],
            "note": f"You should always write in the following language to converse and to write the report {args.language}"},
        )

    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "report writing":         llm_backend,
        "results interpretation": llm_backend,
        "paper refinement":       llm_backend,
    }
    # If a dedicated literature-review backend is provided, apply it just for that phase
    if lite_review_backend is not None and lite_review_backend != "":
        agent_models["literature review"] = lite_review_backend
    RESEARCH_DIR_PATH = str(output_dir) + "/" + run_agent_laboratory.RESEARCH_DIR_PATH
    if parallel_labs:
        lab_utils.remove_figures()
        GLOBAL_AGENTRXIV = run_agent_laboratory.AgentRxiv()
        lab_utils.remove_directory(f"{RESEARCH_DIR_PATH}")
        os.mkdir(os.path.join(".", f"{RESEARCH_DIR_PATH}"))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if not compile_pdf: raise Exception("PDF compilation must be used with agentRxiv!")
        def run_lab(parallel_lab_index):
            time_str = str()
            time_now = time.time()
            for _paper_index in range(num_papers_to_write):
                lab_dir = os.path.join(RESEARCH_DIR_PATH, f"research_dir_lab{parallel_lab_index}_paper{_paper_index}")
                os.mkdir(lab_dir)
                os.mkdir(os.path.join(lab_dir, "src"))
                os.mkdir(os.path.join(lab_dir, "tex"))
                lab_instance = run_agent_laboratory.LaboratoryWorkflow(
                    parallelized=True,
                    research_topic=research_topic,
                    notes=task_notes_LLM,
                    agent_model_backbone=agent_models,
                    human_in_loop_flag=human_in_loop,
                    openai_api_key=api_key,
                    compile_pdf=compile_pdf,
                    num_papers_lit_review=num_papers_lit_review,
                    papersolver_max_steps=papersolver_max_steps,
                    mlesolver_max_steps=mlesolver_max_steps,
                    paper_index=_paper_index,
                    lab_index=parallel_lab_index,
                    except_if_fail=except_if_fail,
                    lab_dir=lab_dir,
                    agentRxiv=True,
                    agentrxiv_papers=args.agentrxiv_papers,
                    # max_steps=2
                )
                if only_literature_review:
                    lab_instance.set_model(lite_review_backend)
                    lab_instance.literature_review()
                else:
                    lab_instance.perform_research()
                time_str += str(time.time() - time_now) + " | "
                with open(f"agent_times_{parallel_lab_index}.txt", "w") as f:
                    f.write(time_str)
                time_now = time.time()

        with ThreadPoolExecutor(max_workers=num_parallel_labs) as executor:
            futures = [executor.submit(run_lab, lab_idx) for lab_idx in range(num_parallel_labs)]
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: print(f"Error in lab: {e}")

        raise NotImplementedError("Todo: implement parallel labs")
    else:
        # remove previous files
        lab_utils.remove_figures()
        if agentRxiv: GLOBAL_AGENTRXIV = run_agent_laboratory.AgentRxiv(lab_index)
        if not agentRxiv:
            lab_utils.remove_directory(f"{RESEARCH_DIR_PATH}")
            os.mkdir(os.path.join(".", f"{RESEARCH_DIR_PATH}"))
        # make src and research directory
        if not os.path.exists("state_saves"): os.mkdir(os.path.join(".", "state_saves"))
        time_str = str()
        time_now = time.time()
        for _paper_index in range(num_papers_to_write):
            lab_direct = f"{RESEARCH_DIR_PATH}/research_dir_{_paper_index}_lab_{lab_index}"
            os.mkdir(os.path.join(lab_direct))
            os.mkdir(os.path.join(f"{lab_direct}", "src"))
            os.mkdir(os.path.join(f"{lab_direct}", "tex"))
        lab = run_agent_laboratory.LaboratoryWorkflow(
            research_topic=research_topic,
                notes=task_notes_LLM,
                agent_model_backbone=agent_models,
                human_in_loop_flag=human_in_loop,
                openai_api_key=api_key,
                compile_pdf=compile_pdf,
                num_papers_lit_review=num_papers_lit_review,
                papersolver_max_steps=papersolver_max_steps,
                mlesolver_max_steps=mlesolver_max_steps,
                paper_index=_paper_index,
                except_if_fail=except_if_fail,
                agentRxiv=False,
                lab_index=lab_index,
                lab_dir=f"{lab_direct}"
            )
        if only_literature_review:
            lab.set_model(lite_review_backend)
            lab.literature_review()
        else:
            lab.perform_research()
            time_str += str(time.time() - time_now) + " | "
            with open(f"agent_times_{lab_index}.txt", "w") as f:
                f.write(time_str)
            time_now = time.time()


def run_agent_laboratory_human_llm(
    topic: str,
    *,
    output_dir: str = None,
    focus_agent: str = "literature_review",  # "all" or "literature_review"
    num_papers_to_write_override: int | None = None,
    num_papers_lit_review_override: int | None = None,
    human_llm: HumanLLM = None,
    human_llm_parameters: Dict = None,
    **kwargs
)-> Tuple[str, List[Dict[str, Any]]]:
    if not output_dir:
        xp_id = (
            SHARED_ID
            + "_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + topic.replace(" ", "_").replace("/", "_")[:50]
        )
        output_dir = "results/tmp_agent_laboratory_human_llm/" + str(xp_id)
        os.makedirs(output_dir, exist_ok=True)
    llm_backend_in = None
    lite_review_backend = None
    if human_llm_parameters:
        # If focusing on a specific phase, set only the relevant backend.
        if (focus_agent or "").lower() == "literature_review":
            lite_review_backend = "humanllm"
        elif (focus_agent or "").lower() == "all":
            llm_backend_in = "humanllm"
            lite_review_backend = "humanllm"

    try:
        import benchmark.agent_lab_import_hook as lab_hook  # installs the hook at interpreter startup
        lab_hook._human_llm_parameters = human_llm_parameters
        lab_hook._human_llm_config = get_default_human_llm_config()
        lab_hook.create_human_llm_from_config = create_human_llm_from_config
        print("[sitecustomize] inference_import_hook loaded")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[sitecustomize] failed to load inference_import_hook:", e)

    run_agent_laboratory(
        topic=topic,
        output_dir=output_dir,
        llm_backend_in=llm_backend_in,
        lite_review_backend=lite_review_backend,
        only_literature_review=(focus_agent or "").lower() == "literature_review",
        num_papers_to_write_override=num_papers_to_write_override,
        num_papers_lit_review_override=num_papers_lit_review_override,
    )

def run_agent_laboratory_survey(
    topic: str,
    intent: Optional[str] = None,
    *,
    llm_backend: str = "o3-mini",
    language: str = "English",
    num_papers_lit_review: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Survey-mode adapter for AgentLaboratory.
    Runs ONLY the literature_review phase, avoids planning/solvers/experiments.
    Returns (report_markdown, turns).
    """
    # Try both import paths: local checkout or vendored
    LabMod = None
    for modname in (
        "agent_laboratory.run_agent_laboratory",
        "ext.agent_laboratory.run_agent_laboratory",
    ):
        try:
            LabMod = importlib.import_module(modname)
            break
        except Exception:
            continue
    if LabMod is None:
        logging.warning("AgentLaboratory not available; returning placeholder review.")
        return f"# Literature Review: {topic}\n\n(AgentLaboratory not installed; placeholder.)", []

    # Build a minimal runtime config (values are strings in their CLI parser)
    cfg = {
        "research_topic": topic,
        "api_key": os.getenv("OPENAI_API_KEY") or "",
        "compile_latex": "False",               # no PDF building in survey mode
        "llm_backend": llm_backend,
        "language": language,
        "num_papers_lit_review": str(num_papers_lit_review),
        "mlesolver_max_steps": "0",             # ensure no code/experiments
        "papersolver_max_steps": "0",
    }

    try:
        lab = LabMod.LaboratoryWorkflow(cfg)
        # Run ONLY literature review. This pulls N papers (arXiv) and summarizes them.
        lab.literature_review()
        review = getattr(lab.phd, "lit_review_sum", None) or ""
        report = f"# Literature Review: {topic}\n\n" + review
        turns: List[Dict[str, Any]] = []  # keep empty; AgentLaboratory doesn't expose fine-grained dialog by default
        return report, turns
    except Exception as e:
        logging.exception("AgentLaboratory literature review failed: %s", e)
        return f"# Literature Review: {topic}\n\n(AgentLaboratory failed: {e})", []


def run_ai_scientist_survey(
    topic: str,
    intent: Optional[str] = None,
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1400,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Survey-mode adapter for AI-Scientist-V2.
    Avoids experiments. Optionally uses their SemanticScholar tool to collect papers,
    then synthesizes a literature review via our LitellmModel (same cost tracking as STORM).
    Returns (report_markdown, turns).
    """
    citations_text = ""
    # 1) Try to use AI-Scientist's Semantic Scholar tool (optional)
    try:
        s2 = None
        for modname in (
            "ai_scientist.tools.semantic_scholar",
            "ext.ai_scientist.tools.semantic_scholar",
        ):
            try:
                s2 = importlib.import_module(modname)
                break
            except Exception:
                continue
        if s2 is not None and hasattr(s2, "SemanticScholarSearchTool"):
            tool = s2.SemanticScholarSearchTool()
            # Keep it light: single query with default k inside the tool
            citations_text = tool.use_tool(query=topic) or ""
    except Exception as e:
        logging.warning("AI-Scientist SemanticScholar tool failed (%s); proceeding without seed citations.", e)

    # 2) Synthesize a literature review via our standard LM wrapper
    lm = _make_openai_lm(model, max_tokens=max_tokens)
    sys_prompt = (
        "You are an expert researcher. Write a concise, neutral literature review on the topic. "
        "Do NOT propose or run experiments, do not ingest datasets, and do not produce code. "
        "Cover relevance, breadth, depth, and novelty. Use section headings."
    )
    user_prompt = f"Topic: {topic}\n\nSeed citations (may be empty):\n{citations_text}"
    try:
        text = lm(messages=[{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt}])[0]
    except Exception as e:
        logging.exception("LM call failed in AI-Scientist survey: %s", e)
        text = f"(LM call failed: {e})"

    report = f"# Literature Review: {topic}\n\n{text}"
    turns: List[Dict[str, Any]] = []  # AI-Scientist survey path aggregates in one shot
    return report, turns

METHODS: Dict[str, Callable[..., Tuple[str, List[Dict[str, Any]]]]] = {
    "Storm": lambda topic, intent=None: run_storm(
        topic,
        **{
            **DEFAULT_PARAMS["Storm"],
            "retriever": ARGS.retriever,
            "storm_path": ARGS.storm_path,
        },
    ),
    "CoStorm": lambda topic, intent=None: run_costorm(
        topic,
        **{
            **DEFAULT_PARAMS["CoStorm"],
            "retriever": ARGS.retriever,
            "total_conv_turn": ARGS.total_conv_turn,
            "storm_path": ARGS.storm_path,
        },
    ),
    "AIScientist": lambda topic, intent=None: run_ai_scientist(
        topic,
        intent,
        **{
            **DEFAULT_PARAMS["AIScientist"],
        },
    ),
    "AgentLaboratory": lambda topic, intent=None: run_agent_laboratory(
        topic,
        intent,
        **{
            **DEFAULT_PARAMS["AgentLaboratory"],
        },
    ),
    "AgentLaboratory_HumanLLM": lambda topic, intent=None: run_agent_laboratory_human_llm(
        topic,
        intent,
        **{
            **DEFAULT_PARAMS["AgentLaboratory_HumanLLM"],
        },
    ),
    "CoStormGraph": lambda topic, intent=None: run_costorm_graph(
        topic, **DEFAULT_PARAMS["CoStormGraph"]
    ),
    "WriteHere": lambda topic, intent: run_writehere(
        topic,
        intent,
        **{
            **DEFAULT_PARAMS["WriteHere"],
            "writehere_model": ARGS.writehere_model,
            "engine_backend": ARGS.retriever,
            "writehere_path": ARGS.writehere_path,
        },
    ),
}


# ────────────────────────── CLI PARSING ──────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("benchmark.py – unified STORM / WriteHERE benchmark")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_LOADERS),
        default=list(DATASET_LOADERS),
        help="Datasets to benchmark",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="Generation methods to benchmark",
    )
    p.add_argument(
        "--dea-root",
        type=Path,
        default=Path.home()
        / "CollabFunctionsGPTCreator"
        / "env"
        / "IR_CPS_TechSynthesis"
        / "document_embedding_analysis",
        help="Root folder of DEA (contains output/latex/*.json)",
    )
    p.add_argument(
        "--storm-path", type=Path, help="Existing clone of stanford-oval/storm"
    )
    p.add_argument(
        "--costorm-graph-path", type=Path, help="Existing clone of doxav/storm-swarm"
    )
    p.add_argument(
        "--ai-scientist-path", type=Path, help="Existing clone of AI-Scientist"
    )
    p.add_argument(
        "--agent-laboratory-path", type=Path, help="Existing clone of Agent Laboratory"
    )
    p.add_argument(
        "--total-conv-turn",
        type=int,
        default=3,
        help="Number of conversational turns for Co-STORM",
    )
    p.add_argument(
        "--writehere-path", type=Path, help="Existing clone of principia-ai/WriteHERE"
    )
    p.add_argument(
        "--writehere-model", default="gpt-5-nano", help="Model name for WriteHERE"
    )
    p.add_argument(
        "--writehere-backend",
        default="openai",
        choices=["openai", "azure", "anthropic"],
        help="Backend for WriteHERE LLM calls",
    )
    p.add_argument(
        "--retriever",
        default="searxng",
        choices=[
            "bing",
            "you",
            "brave",
            "serper",
            "duckduckgo",
            "tavily",
            "searxng",
            "azure_ai_search",
        ],
        help="Retriever/search engine to use across all methods by default.",
    )
    return p


# Keep ARGS global so lambdas in METHODS can access
ARGS: argparse.Namespace

# Unique ID for this run, used in output filenames
SHARED_ID = str(get_last_id(RESULT_DIR) + 1).zfill(7)


# ────────────────────────── MAIN BENCHMARK LOOP ──────────────────────────
def main() -> None:
    global STORM_DIR, SWARM_DIR, WRITEHERE_DIR, ARGS, SWARM_GIT, STORM_GIT, STORM_GIT_COMMIT, AISCIENTIST_DIR, \
    AISCIENTIST_GIT, AISCIENTIST_GIT_COMMIT, AGENTLABORATORY_DIR, AGENTLAB_GIT, AGENTLAB_GIT_COMMIT
    ARGS = _build_parser().parse_args()

    # Lazy clone / path-inject external repos
    # if {"Storm", "CoStorm", "CoStormGraph"} & set(ARGS.methods):
    #     STORM_DIR = ARGS.storm_path or STORM_DIR or SOURCES_DIR / "storm"
    #     clone_repo(STORM_DIR, STORM_GIT, commit=STORM_GIT_COMMIT)
    #     sys.path.insert(0, str(STORM_DIR))

    # if "CoStormGraph" in ARGS.methods:
    #     SWARM_DIR = ARGS.costorm_graph_path or SWARM_DIR or SOURCES_DIR / "storm-swarm"
    #     clone_repo(SWARM_DIR, SWARM_GIT)
    #     sys.path.insert(0, str(SWARM_DIR))

    # if "WriteHere" in ARGS.methods:
    #     WRITEHERE_DIR = (
    #         ARGS.writehere_path or WRITEHERE_DIR or SOURCES_DIR / "WriteHERE"
    #     )
    #     clone_repo(WRITEHERE_DIR, WRITEHERE_GIT, commit=WRITEHERE_GIT_COMMIT)
    #     sys.path.insert(0, str(WRITEHERE_DIR))

    # if "AIScientist" in ARGS.methods:
    #     AISCIENTIST_DIR = ARGS.ai_scientist_path or AISCIENTIST_DIR or SOURCES_DIR / "ai_scientist_v2"
    #     clone_repo(AISCIENTIST_DIR, AISCIENTIST_GIT, commit=AISCIENTIST_GIT_COMMIT)
    #     sys.path.insert(0, str(AISCIENTIST_DIR))

    if "AgentLaboratory" in ARGS.methods:
        AGENTLABORATORY_DIR = ARGS.agent_laboratory_path or AGENTLABORATORY_DIR or SOURCES_DIR / "agent_laboratory"
        clone_repo(AGENTLABORATORY_DIR, AGENTLAB_GIT, commit=AGENTLAB_GIT_COMMIT)
        sys.path.insert(0, str(AGENTLABORATORY_DIR))

    rows: List[Dict[str, Any]] = []

    for dset in ARGS.datasets:
        logging.info("→ Dataset: %s", dset)
        loader = DATASET_LOADERS[dset]
        iterable = loader(ARGS.dea_root) if dset == "DEA" else loader()

        for method in ARGS.methods:
            if method != "AgentLaboratory_HumanLLM":
                continue
            logging.info("   → Method: %s", method)
            bench_dir = RESULT_DIR / f"{dset}_{method}"
            bench_dir.mkdir(parents=True, exist_ok=True)

            for topic, intent, solution in iterable:
                start = time.time()
                report, turns = METHODS[method](topic, intent)
                metrics = _evaluate_and_dump(
                    topic=topic,
                    report=report,
                    turns=turns,
                    method=method,
                    dataset=dset,
                    out_dir=bench_dir,
                    solution=solution if dset == "DEA" else None,
                )
                elapsed = round(time.time() - start, 2)

                flat: Dict[str, Any] = {
                    "dataset": dset,
                    "method": method,
                    "topic": topic,
                    "intent": intent,
                    "seconds": elapsed,
                }

                # flatten metrics one level
                def _flatten(prefix: str, obj: Dict[str, Any]):
                    for k, v in obj.items():
                        key = f"{prefix}{k}"
                        if isinstance(v, dict):
                            yield from _flatten(f"{key}.", v)
                        else:
                            yield key, v

                for k, v in _flatten("", metrics):
                    flat[k] = v

                rows.append(flat)

    if rows:
        csv_path = RESULT_DIR / "all_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    logging.info(
        "Benchmark finished – aggregated CSV at %s", RESULT_DIR / "all_results.csv"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
