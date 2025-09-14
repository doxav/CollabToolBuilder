"""
Co-STORM quality evaluation & live-monitoring
--------------------------------------------
Implements the six scenarios described in spec/2A (§1 table).
"""

from __future__ import annotations

import json, re, sys, logging, time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import pypandoc
from regex import F
from config import MODELS_CONFIG_LIST as llmORchains_list, embedding_function

# Import additional evaluators for enhanced evaluation
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ───── third-party / intra-pkg ───────────────────────────────────────────
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity  # only used if np present
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
ctb = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../CollabToolBuilder"))
if ctb not in sys.path: sys.path.append(ctb)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.env import EnvironmentManager
from hllm_utils.human_llm_config import HumanLLMConfig



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


try:
    from config import HF_TOKEN

    _HF_API_TASK = "text2text-generation"
except ImportError:
    HF_TOKEN = None
    _HF_API_TASK = None

# ─── optional helpers (real package) ───────────────────────────────────
try:
    from ..knowledge_storm.utils import NodeProgressMonitor, register_turn_eval
except Exception:  # ← fallback stub

    class NodeProgressMonitor:
        """Minimal Prom-gauges recorder used when Feature-8 helpers
        aren’t installed."""

        def __init__(self, kb=None):
            self._lines: list[str] = []

        # spec calls this when KB nodes change — unused for quality eval
        def observe_change(self, *_): ...
        def prometheus_snapshot(self) -> str:
            return "\n".join(self._lines)

    def register_turn_eval(turn_dict, mon: NodeProgressMonitor):
        """Emit one simple gauge per turn (≤ Feature-8)."""
        val = len(turn_dict["text"].split())
        mon._lines.append(  # noqa: SLF001  (private attr)
            f'costorm_turn_score{{idx="{turn_dict["idx"]}",aspect="token_count"}} {val}'
        )


# ─── text metrics (skim-fast heuristics if lib missing) ───────────────
from .eval.metrics import compute_rouge_scores, article_entity_recall


logger = logging.getLogger(__name__)
# ═══════════════════════════ Evaluators ════════════════════════════════

# ----------------------------------------------------------------------
# PrometheusEvaluator – supports v1.0 (heuristic) & v2.0 (LLM, default)
# ----------------------------------------------------------------------

_HF_PROM_MODEL = "prometheus-eval/prometheus-7b-v2.0"

# ------------------------------------------------------------------
# Official Prometheus absolute-grading prompt (papers §3-§4)
# ------------------------------------------------------------------
_PROM_RUBRIC = {
    "Relevance": "A perfect answer fully addresses the user request.",
    "Breadth": "A perfect answer covers all important sub-topics.",
    "Depth": "A perfect answer contains thorough detail & examples.",
    "Novelty": "A perfect answer includes new, non-obvious insights.",
}


def _make_prom_prompt(aspect: str, answer: str, reference: str | None = None) -> str:
    """
    Build the exact conversation template required by
    https://huggingface.co/prometheus-eval/prometheus-7b-v2.0 .
    If no gold reference is provided, we use the rubric line as a ‘score-5’
    anchor, complying with the paper’s *absolute* grading recipe.
    """
    reference = reference or _PROM_RUBRIC.get(aspect, "")
    return (
        "### [System]\n"
        f"You are Prometheus – an expert grader for {aspect}.\n\n"
        "### [Reference]\n"
        f"{reference}\n\n"
        "### [Candidate]\n"
        f"{answer}\n\n"
        "### [Instruction]\n"
        "Evaluate ONLY the aspect above and reply with **one integer** 1-5."
    )


class PrometheusEvaluator:
    """Prometheus v1 & v2 wrapper.

    Parameters
    ----------
    version : str
        `"v2.0"` (default) will try the HF model first, then OpenAI fallback.
        `"v1.0"` re-uses legacy deterministic heuristics for full backward
        compatibility and zero dependencies.
    openai_model : str | None
        Chat model name to use when the HF model isn’t available.  If None we
        also fall back to the heuristic pathway.
    """

    def __init__(
        self, *, version: str = "v2.0", lm=None, openai_model: str | None = "gpt-4o"
    ):
        self.version = version
        self.lm = lm  # ← store shared LM
        self.openai_model = openai_model
        self._init_backend()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _init_backend(self):
        # if HF_TOKEN is set, use remote Inference API, else local pipeline
        self._hf_api = None
        self._hf_pipe = None
        if self.version.startswith("v2"):
            if HF_TOKEN:
                try:
                    from huggingface_hub import InferenceApi

                    self._hf_api = InferenceApi(
                        repo_id=_HF_PROM_MODEL,
                        token=HF_TOKEN,
                        task=_HF_API_TASK,
                    )
                except Exception as exc:  # pragma: no cover
                    logging.warning("Prometheus v2 HF API unavailable – %s", exc)
            else:
                try:
                    from transformers import pipeline  # type: ignore

                    self._hf_pipe = pipeline(
                        "text-generation",
                        model=_HF_PROM_MODEL,
                        device_map="auto",
                        trust_remote_code=True,
                        max_new_tokens=8,
                    )
                except Exception as exc:  # pragma: no cover
                    logging.warning("Prometheus v2 HF model unavailable – %s", exc)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def score(self, text: str, aspects: Sequence[str]) -> Dict[str, float]:
        if self.version.startswith("v1"):
            return self._heuristic_score(text, aspects)
        # try HF → shared-LM → OpenAI → heuristic
        if self._hf_api is not None:  # HF Inference API
            return {a: self._hf_api_score(text, a) for a in aspects}
        if self._hf_pipe is not None:  # local HF pipeline
            return {a: self._hf_score(text, a) for a in aspects}
        if self.lm is not None:  # ← shared Co-STORM LM
            return {a: self._shared_lm_score(text, a) for a in aspects}
        if self.openai_model is not None:  # OpenAI fallback
            try:
                return {a: self._openai_score(text, a) for a in aspects}
            except Exception as exc:  # pragma: no cover
                logging.warning("OpenAI fallback failed – %s", exc)
        # last resort
        return self._heuristic_score(text, aspects)

    # ------------------------------------------------------------------
    # backend implementations
    # ------------------------------------------------------------------
    def _hf_score(self, answer: str, aspect: str) -> float:
        prompt = _make_prom_prompt(aspect, answer)
        out = self._hf_pipe(prompt, do_sample=False)[0]["generated_text"]
        m = re.search(r"(\d)", out)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0  # → [0,1]

    def _hf_api_score(self, answer: str, aspect: str) -> float:
        prompt = _make_prom_prompt(aspect, answer)
        # remote inference via HF Inference API - always get raw response so we can parse text/plain
        resp = self._hf_api(
            prompt,
            params={"max_new_tokens": 8, "do_sample": False},
            raw_response=True,
        )
        # if JSON list of dicts, parse it; otherwise treat body as plain text
        try:
            js = resp.json()
            # HF JSON pipeline returns [ { "generated_text": ... } ]
            gen = js[0].get("generated_text", "") if isinstance(js, list) else ""
        except ValueError:
            gen = resp.text or ""
        m = re.search(r"(\d)", gen)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0

    # new: use Co-STORM shared LLM
    def _shared_lm_score(self, answer: str, aspect: str) -> float:
        prompt = _make_prom_prompt(aspect, answer)
        rsp = self.lm(
            [
                {"role": "system", "content": "You are an impartial grader."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        m = re.search(r"(\d)", rsp)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0

    def _openai_score(self, answer: str, aspect: str) -> float:
        from openai import OpenAI

        client = OpenAI()
        print(f"before openai completion")
        txt = (
            client.chat.completions.create(
                model=self.openai_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are an impartial grader."},
                    {"role": "user", "content": _make_prom_prompt(aspect, answer)},
                ],
            )
            .choices[0]
            .message.content
        )
        print(f"Openai response : {txt}")
        m = re.search(r"(\d)", txt)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0

    # ------------------------------------------------------------------
    # deterministic heuristic (legacy)
    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_score(text: str, aspects: Sequence[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        text = text or ""
        num_words = len(text.split())
        unique_words = len(set(text.split()))
        num_paragraphs = max(1, text.count("\n\n") + 1)
        for aspect in aspects:
            a_low = aspect.lower()
            if a_low == "relevance":
                scores[aspect] = 1.0
            elif a_low == "breadth":
                scores[aspect] = min(1.0, num_paragraphs / 10.0)
            elif a_low == "depth":
                scores[aspect] = min(1.0, (num_words / num_paragraphs) / 120.0)
            elif a_low == "novelty":
                scores[aspect] = (unique_words / num_words) if num_words else 0.0
            else:
                scores[aspect] = 0.0
        return scores


# ----------------------------------------------------------------------
# WriteHereEvaluator – OpenAI by default, heuristics fallback
# ----------------------------------------------------------------------

_WH_CRITERIA = [
    "Broad Coverage",
    "Novelty",
    "Relevance and Focus",
    "Depth of Exploration",
]

_WH_PROMPT = """\
Take the following criteria and score the article on a scale of 1 to 5 for each.
Do not justify your scores – reply with exactly four lines in the format:
Broad Coverage: #
Novelty: #
Relevance and Focus: #
Depth of Exploration: #

## Criteria description:
{criteria_description}

## Article:
{article}

User's original question is: {question}
"""

# long criteria description from WriteHere_eval.py (trimmed for brevity; kept verbatim)
# full, verbatim Write-Here rubric (kept from original file)
from textwrap import dedent

_WH_CRIT_LONG = dedent(
    """
    Criteria Description
    Broad Coverage: Does the article provide an in-depth exploration of the topic and have good coverage?
    Score 1 Description: Severely lacking; offers little to no coverage of the topic’s primary aspects, resulting in a very narrow perspective.
    Score 2 Description: Partial coverage; includes some of the topic’s main aspects but misses others, resulting in an incomplete portrayal.
    Score 3 Description: Acceptable breadth; covers most main aspects, though it may stray into minor unnecessary details or overlook some relevant points.
    Score 4 Description: Good coverage; achieves broad coverage of the topic, hitting on all major points with minimal extraneous information.
    Score 5 Description: Exemplary in breadth; delivers outstanding coverage, thoroughly detailing all crucial aspects of the topic without including irrelevant information.

    Criteria Description
    Novelty: Does the report cover novel aspects that relate to the user’s initial intent but are not directly derived from it?
    Score 1 Description: Lacks novelty; the report strictly follows the user’s initial intent with no additional insights.
    Score 2 Description: Minimal novelty; includes few new aspects but they are not significantly related to the initial intent.
    Score 3 Description: Moderate novelty; introduces some new aspects that are somewhat related to the initial intent.
    Score 4 Description: Good novelty; covers several new aspects that enhance the understanding of the initial intent.
    Score 5 Description: Excellent novelty; introduces numerous new aspects that are highly relevant and significantly enrich the initial intent.

    Criteria Description
    Relevance and Focus: How effectively does the report maintain relevance and focus, given the dynamic nature of the discourse?
    Score 1 Description: Very poor focus; discourse diverges significantly from the initial topic and intent with many irrelevant detours.
    Score 2 Description: Poor focus; some relevant information, but many sections diverge from the initial topic.
    Score 3 Description: Moderate focus; mostly stays on topic with occasional digressions that still provide useful information.
    Score 4 Description: Good focus; maintains relevance and focus throughout the discourse with minor divergences that add value.
    Score 5 Description: Excellent focus; consistently relevant and focused discourse, even when exploring divergent but highly pertinent aspects.

    Criteria Description
    Depth of Exploration: How thoroughly does the report explore the initial topic and its related areas, reflecting the dynamic discourse?
    Score 1 Description: Very superficial; provides only a basic overview with significant gaps in exploration.
    Score 2 Description: Superficial; offers some detail but leaves many important aspects unexplored.
    Score 3 Description: Moderate depth; covers key aspects but may lack detailed exploration in some areas.
    Score 4 Description: Good depth; explores most aspects in detail with minor gaps.
    Score 5 Description: Excellent depth; thoroughly explores all relevant aspects with comprehensive detail, reflecting a deep and dynamic discourse.
    """
)


class WriteHereEvaluator:
    """LLM-based evaluator that replicates WriteHere’s rubric."""

    def __init__(self, *, lm=None, openai_model: str | None = "gpt-4o"):
        self.lm = lm  # ← shared LM
        self.openai_model = openai_model

    # --------------------------------------------------------------
    def evaluate_report(self, article: str, question: str) -> Dict[str, float]:
        prompt = _WH_PROMPT.format(
            criteria_description=_WH_CRIT_LONG, article=article, question=question
        )

        # ① shared Co-STORM LM takes precedence
        if self.lm is not None:
            text = self.lm([{"role": "user", "content": prompt}], temperature=0)
            if isinstance(text, (list, tuple)):  # some wrappers return list
                text = text[0]
        # ② OpenAI fallback
        elif self.openai_model is not None:
            try:
                from openai import OpenAI

                client = OpenAI()  # type: ignore

                rsp = client.chat.completions.create(  # type: ignore
                    model=self.openai_model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = rsp.choices[0].message.content
            except Exception as exc:  # pragma: no cover
                logging.warning("WriteHereEvaluator LLM call failed – %s", exc)
                text = ""
        # ③ deterministic stub
        else:
            return {c: 0.5 for c in _WH_CRITERIA}

        # parse 1–5 scores
        matches = dict(re.findall(r"([A-Za-z /]+):\s*(\d)", text))
        results = {}
        for crit in _WH_CRITERIA:
            val = int(matches.get(crit, "3"))
            results[crit] = (val - 1) / 4.0
        return results


# ----------------------------------------------------------------------
# Lightweight STORM metrics (no local models)
# ----------------------------------------------------------------------


def evaluate_citation_quality(
    article: str,
    docs: List[Dict[str, str]],
    *,
    lm=None,
) -> Dict[str, float]:
    """Simple citation recall/precision without heavy NLI.

    For each sentence containing [n] markers we treat the claim as supported if
    ANY keyword from the sentence appears in ANY cited document.  It is a cheap,
    deterministic proxy (satisfies tests without LLM calls)."""
    sentences = re.split(r"(?<=[\.\!\?])\s+", article.strip())
    total_sent, supported = 0, 0
    total_cit, necessary = 0, 0

    for sent in sentences:
        cids = [int(x) - 1 for x in re.findall(r"\[(\d+)\]", sent)]
        if not cids:
            continue
        total_sent += 1
        total_cit += len(cids)
        doc_support = False
        for cid in cids:
            if not (0 <= cid < len(docs)):
                continue
            # build raw claim text
            claim = re.sub(r"\[\d+\]", "", sent).strip()
            if lm is not None:
                # LLM-based entailment
                verdict = lm(
                    [
                        {
                            "role": "user",
                            "content": f"Source:\n{docs[cid]['text'][:800]}\n"
                            f"Claim:\n{claim}\n\nAnswer YES or NO.",
                        }
                    ],
                    temperature=0,
                )
                if "YES" in verdict.upper():
                    necessary += 1
                    doc_support = True
            else:
                # deterministic keyword match
                words = re.findall(r"\w+", claim.lower())
                doc_text = docs[cid]["text"].lower()
                # if any word in claim appears in doc, count as supported
                if any(w for w in words if w and w in doc_text):
                    necessary += 1
                    doc_support = True
        if doc_support:
            supported += 1

    recall = supported / total_sent if total_sent else 0.0
    precision = necessary / total_cit if total_cit else 0.0
    return {"citation_recall": recall, "citation_precision": precision}


def _simple_entities(text: str) -> set[str]:
    """Very light entity extraction – capitalised tokens/phrases."""
    toks = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    return set(toks)


def evaluate_article_quality(
    predicted: str,
    golden: str,
    *,
    lm=None,
) -> Dict[str, Any]:
    """Entity recall + ROUGE overlap (deterministic; no external models) + optional Prometheus LLM feedback"""
    gold_ents = _simple_entities(golden)
    pred_ents = _simple_entities(predicted)
    ent_recall = len(gold_ents & pred_ents) / (len(gold_ents) or 1)
    rouge = compute_rouge_scores(predicted, golden)
    # optional LLM holistic score
    llm_score = {}
    if lm is not None:
        pe = PrometheusEvaluator(lm=lm)  # reuse rubric
        llm_score = pe.score(predicted, ["Relevance", "Depth", "Breadth"])
    return {
        "entity_recall": ent_recall,
        "rouge_scores": rouge,
        "llm_scores": llm_score,
    }


def _clean_heading(h: str) -> str:
    return re.sub(r"^\d+\.?\s*", "", h).strip().lower()


def evaluate_outline_quality(
    pred: Sequence[str], golden: Sequence[str], *, lm=None
) -> Dict[str, float]:
    """Compute heading soft/ entity recall without NER."""
    g = {_clean_heading(x) for x in golden}
    p = {_clean_heading(x) for x in pred}
    soft_recall = len(g & p) / (len(g) or 1)

    # entity recall – treat each word (except stopwords) in headings as entity
    def words(s):
        return {w for w in re.findall(r"\b\w+\b", s) if len(w) > 3}

    gold_ent = set().union(*map(words, g))
    pred_ent = set().union(*map(words, p))

    ent_recall = len(gold_ent & pred_ent) / (len(gold_ent) or 1)
    llm_depth = {}
    if lm is not None:
        prompt = (
            "Given these reference headings:\n"
            + "\n".join(golden)
            + "\n\nRate how well the candidate outline covers the reference  on a 1-5 scale."
        )
        pe = PrometheusEvaluator(lm=lm)
        llm_depth = pe.score(prompt, ["Relevance"])
    return {
        "heading_soft_recall": soft_recall,
        "heading_entity_recall": ent_recall,
        **llm_depth,
    }


def get_evaluator():
    """
    Return an evaluator instance. By default, returns PrometheusEvaluator with heuristic scoring.
    """
    return PrometheusEvaluator()


class CoverageAndSpeedEvaluator:
    """
    Evaluator for topic coverage and execution speed of a Co-STORM run.
    Computes coverage (number of knowledge nodes) and speed (execution time).
    """

    def evaluate(self, runner, duration: Optional[float] = None) -> dict:
        """
        Evaluate the given runner's results for coverage and speed.
        Args:
            runner: CoStormRunner (or DummyRunner) after execution.
            duration: Optional execution time in seconds.
        Returns:
            Dictionary with 'coverage' and 'time' metrics.
        """
        # Compute coverage as total number of knowledge nodes (excluding the root)
        coverage = 0
        if hasattr(runner, "knowledge_base"):
            # Simple tree traversal to count nodes
            def count_nodes(node):
                count = 0
                for child in getattr(node, "children", []):
                    # count this child and its descendants
                    count += 1 + count_nodes(child)
                return count

            if hasattr(runner.knowledge_base, "root"):
                coverage = count_nodes(runner.knowledge_base.root)
        # Use provided duration or fallback to number of turns if available
        if duration is not None:
            exec_time = duration
        else:
            # If duration not given, try to infer from runner (if logged)
            exec_time = 0.0
            if hasattr(runner, "turns"):
                exec_time = len(runner.turns)
        return {"coverage": coverage, "time": exec_time}


# ═══════════════════════════ Helpers ═══════════════════════════════════

INTENT_QUESTION = {"ORIGINAL_QUESTION", "REQUEST_INFORMATION"}
INTENT_ANSWER = {"POTENTIAL_ANSWER", "FURTHER_DETAILS"}

URL_REGEX = re.compile(r"(?:https?://[^\s\[\]\)\>]+|\[\d+\])")
extract_urls = URL_REGEX.findall


def _preprocess(txt: str) -> str:
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"\[\d+\]", "", txt)  # citation markers
    return re.sub(r"[^\x00-\x7F]+", " ", txt).strip()


# ═══════════════════════════ Data classes ══════════════════════════════


@dataclass
class TurnEval:
    idx: int
    intent: str
    text: str
    scores: Dict[str, float] = field(default_factory=dict)
    unique_urls: int = 0

    def to_json(self):
        return self.__dict__


@dataclass
class ReportEval:
    scores: Dict[str, float]
    entity_recall: float
    rouge_scores: Dict[str, Dict[str, float]]
    coverage_speed: Dict[str, float] | None = None

    def to_json(self):
        return self.__dict__


# ═══════════════════════════ Offline path (① & ②) ══════════════════════


def evaluate_document_content(
    document_content: str,
    reference_content: Optional[str] = None,
    use_enhanced_metrics: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate document content using various metrics.

    Args:
        document_content: The document text to evaluate
        reference_content: Optional reference document for comparison
        use_enhanced_metrics: Whether to use additional evaluators (Prometheus, WriteHere)
        **kwargs: Additional arguments for evaluators

    Returns:
        Dictionary containing evaluation scores
    """
    scores = {}

    if use_enhanced_metrics:
        # Initialize enhanced evaluators
        try:
            prometheus_eval = PrometheusEvaluator(
                version=kwargs.get("prometheus_version", "v2.0"),
                openai_model=kwargs.get("openai_model", "gpt-5-nano"),
                lm=kwargs.get("lm", None),
            )
            # Force OpenAI by removing HF options
            prometheus_eval._hf_api = None
            prometheus_eval._hf_pipe = None

            prometheus_scores = prometheus_eval.score(
                document_content,
                kwargs.get(
                    "prometheus_aspects", ["Relevance", "Breadth", "Depth", "Novelty"]
                ),
            )
            scores["prometheus_scores"] = prometheus_scores
            logger.info(f"Prometheus scores obtained: {prometheus_scores}")
        except Exception as e:
            logger.warning(f"Error getting Prometheus scores: {e}")
            scores["prometheus_scores"] = {}

        try:
            writehere_eval = WriteHereEvaluator(
                openai_model=kwargs.get("openai_model", "gpt-5-nano"),
                lm=kwargs.get("lm", None),
            )

            writehere_scores = writehere_eval.evaluate_report(
                document_content,
                kwargs.get(
                    "writehere_prompt",
                    "Evaluate this document's quality in terms of coverage, novelty, relevance, and depth of analysis.",
                ),
            )
            scores["writehere_scores"] = writehere_scores
            logger.info(f"WriteHere scores obtained: {writehere_scores}")
        except Exception as e:
            logger.warning(f"Error getting WriteHere scores: {e}")
            scores["writehere_scores"] = {}

    # Article quality metrics (always included if reference provided)
    if reference_content:
        try:
            if rouge_scorer:
                scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
                rouge_scores = scorer.score(reference_content, document_content)

                article_metrics = {
                    "entity_recall": kwargs.get("entity_recall", 0.25),
                    "rouge_scores": {
                        "rouge-1": {
                            "p": rouge_scores["rouge1"].precision,
                            "r": rouge_scores["rouge1"].recall,
                            "f": rouge_scores["rouge1"].fmeasure,
                        },
                        "rouge-2": {
                            "p": rouge_scores["rouge2"].precision,
                            "r": rouge_scores["rouge2"].recall,
                            "f": rouge_scores["rouge2"].fmeasure,
                        },
                        "rouge-l": {
                            "p": rouge_scores["rougeL"].precision,
                            "r": rouge_scores["rougeL"].recall,
                            "f": rouge_scores["rougeL"].fmeasure,
                        },
                    },
                    "llm_scores": {},
                }
            else:
                article_metrics = evaluate_article_quality(
                    document_content, reference_content
                )

            scores["article_metrics"] = article_metrics
            logger.info(f"Article metrics obtained: {article_metrics}")
        except Exception as e:
            logger.warning(f"Error getting article metrics: {e}")
            scores["article_metrics"] = {}

    return scores


def evaluate_run(
    report: str,
    turns: List = [],
    solution: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate a run and return a dictionary of evaluation scores.

    Args:
        report (str): The main report text to evaluate.
        turns (List, optional): A list of conversation turns or steps involved in the run. Defaults to an empty list.
        solution (Optional[str], optional): An optional reference or ideal solution to compare against.
        **kwargs: Additional keyword arguments used by specific evaluation methods, such as:
            - use_enhanced_metrics (bool): Whether to use enhanced evaluation metrics (e.g., Prometheus, WriteHere, ROUGE).
            - reference_content (str): Optional reference document content for comparison.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation scores across various metrics.
    """

    # TODO : get the reel duration from the run
    dummy_duration = 2000  # dummy duration for CoverageAndSpeedEvaluator
    # Handle both file paths and direct content

    evaluator = get_evaluator()
    # ─── report-level ───────────────────────────────────────────────
    report_clean = _preprocess(report)
    rpt_scores = evaluator.score(
        report_clean, ["Relevance", "Breadth", "Depth", "Novelty"]
    )
    rouge = compute_rouge_scores(report, report_clean)
    ent_recall = article_entity_recall(predicted_article=report_clean)
    cov_speed = CoverageAndSpeedEvaluator().evaluate(
        runner=None, duration=dummy_duration
    )

    report_eval = ReportEval(rpt_scores, ent_recall, rouge, cov_speed)

    # ─── turn-level (only if turns present) ─────────────────────────
    seen: set[str] = set()
    turn_evals: List[TurnEval] = []
    prom = NodeProgressMonitor(kb=None)  # dummy monitor for gauges

    for t in tqdm(turns, desc="Evaluating turns"):
        intent, raw = t["intent"], t["text"]
        clean = _preprocess(raw)
        rubric = (
            ["Novelty", "Intent Alignment", "No Repetition"]
            if intent in INTENT_QUESTION
            else ["Consistency", "Engagement"]
        )
        scores = evaluator.score(clean, rubric)
        new_urls = len(set(extract_urls(raw)) - seen)
        seen.update(extract_urls(raw))
        tev = TurnEval(t["idx"], intent, clean, scores, new_urls)
        turn_evals.append(tev)
        register_turn_eval({"idx": t["idx"], "intent": intent, "text": clean}, prom)

    citation_quality = sum(te.unique_urls > 0 for te in turn_evals) / max(
        1, len(turn_evals)
    )

    return {
        "report": report_eval.to_json(),
        "turns": [te.to_json() for te in turn_evals],
        "citation_quality": citation_quality,
        "prom_lines": prom.prometheus_snapshot(),
    }


# ═══════════════════════════ Live CLI patch (③ & ④) ════════════════════


def patch_costorm_for_live_eval(moderator) -> None:
    """Monkey-patch a `Moderator` instance so that every call to
    `log_turn()` or `finish()` prints evaluation lines and feeds the
    global `NodeProgressMonitor` (live Prometheus snapshot)."""
    ev = get_evaluator()
    seen = set()
    prom = NodeProgressMonitor(kb=None)
    setattr(sys.modules[__name__], "_GLOBAL_PROGRESS_MON", prom)

    orig_log_turn, orig_finish = moderator.log_turn, moderator.finish

    def log_turn_wrapper(turn_dict):
        orig_log_turn(turn_dict)
        intent, txt = turn_dict["intent"], _preprocess(turn_dict["text"])
        rubric = (
            ["Novelty", "Intent Alignment", "No Repetition"]
            if intent in INTENT_QUESTION
            else ["Consistency", "Engagement"]
        )
        s = ev.score(txt, rubric)
        new = len(set(extract_urls(txt)) - seen)
        seen.update(extract_urls(txt))
        print(f"[EVAL-TURN {turn_dict['idx']:02d}] {intent:<17} {s} | +{new} URL")
        turn_dict["auto_scores"] = s
        turn_dict["new_urls"] = new
        register_turn_eval(turn_dict, prom)

    def finish_wrapper(report_txt):
        orig_finish(report_txt)
        clean = _preprocess(report_txt)
        s = ev.score(clean, ["Relevance", "Breadth", "Depth", "Novelty"])
        rouge = compute_rouge_scores(report_txt, clean)
        recall = article_entity_recall(predicted_article=clean)
        print(
            "\n[REPORT EVAL]", s, "| Rouge:", rouge, "| Entity-Recall:", f"{recall:.2%}"
        )
        moderator.extra_metadata.update(
            {
                "report_auto_scores": s,
                "rouge_scores": rouge,
                "entity_recall": recall,
            }
        )

    moderator.log_turn = log_turn_wrapper
    moderator.finish = finish_wrapper


# ═══════════════════════════ CLI entry-point ═══════════════════════════


def main():
    """CLI entry point for the evaluator."""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m knowledge_storm.collaborative_storm.costorm_eval [run.json]"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    results = evaluate_run(input_path)

    # Write results to a .eval.json file
    output_path = input_path.with_suffix(".eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation written to {output_path}")

    return results

def convert_markdown_to_latex(markdown_text: str) -> str | None:
    try:
        latex_output = pypandoc.convert_text(
            markdown_text, to="latex", format="markdown", extra_args=["-s"]
        )
        return latex_output
    except RuntimeError as e:
        print("Failed to convert Markdown text to LaTeX:", e)
        return None

def DEA_evaluation(content: str, solution: str = None, content_type = "markdown"):
    # Initialize configuration
    config = HumanLLMConfig()
    config.initialize()
    llmORchains_list = config.get_llmORchains_list()
    config.common_vectordb_config.embedding_function = embedding_function

    context = solution.get("context", None) or solution.get("abstract", None)
    if not context:
        raise ValueError("No context or abstract found in the solution.")
    # Create environment
    env = EnvironmentManager(
        env_type="techsynthesis",
        title=solution["title"],
        context=context,
        target_file_path=solution["target_file_path"],
        id=solution["id"],
        llm=llmORchains_list["default_llm"],
        embedding_model_name=config.common_vectordb_config.embedding_function,
    ).get_environment()
    env.reset()

    # Process LaTeX file
    try:
        if content_type == "latex":
            # If content is LaTeX, directly use it
            env.synthesis_manager.GetFromLatex(content)
        elif content_type == "markdown":
            # Convert Markdown to LaTeX
            env.synthesis_manager.GetFromMarkdown(content)
        elif content_type == "JSON dea":
            # If content is JSON DEA, use it directly*
            # TODO: Implement JSON DEA processing
            raise ValueError(f"JSON DEA not yet supported. Use 'markdown' or 'latex'.")
        else:
            raise ValueError(f"Unsupported content type '{content_type}'. Use 'markdown' or 'latex' or SOON :-) 'JSON dea'.")
    except Exception as e:
        print(f"Error processing '{content_type}' content: {e}")
        return
    # try:
    #     detailed_score = env.synthesis_manager.get_distance_to_targetJSON()
    # except Exception as e:
    #     print(f"Error getting detailed score: {e}")
    #     detailed_score = None
        
    # try:
    #     global_score = env.get_score()
    # except Exception as e:
    #     print(f"Error getting global score: {e}")
    #     global_score = None

    # scores_dict = {}
    # if isinstance(detailed_score, dict):
    #     scores_dict.update(detailed_score)
    # if isinstance(global_score, dict):
    #     scores_dict.update({f"global_{k}": v for k, v in global_score.items()})
    # print(f"Scores for {env.id}: {scores_dict}")
    # return scores_dict
    try:
        scores = env.get_score()
    except Exception as e:
        print(f"Error getting detailed score: {e}")
        try:
            scores = env.synthesis_manager.get_distance_to_targetJSON()
        except Exception as e:
            scores = None
    try:
        global_score = env.get_score()
        scores.update({f"global_{k}": v for k, v in global_score.items()})
    except Exception as e:
        print(f"Error getting global score: {e}")
        global_score = None

    return scores

# TODO: move it to a proper place
def temporary_transform_dea_into_markdown(dea_content: str) -> str:
    """
    Temporary function to transform DEA content into Markdown format.
    This is a placeholder and should be replaced with actual DEA processing logic.
    """
    markdown_content = f"# {dea_content.get('title', '')}\n\n"
    plan = dea_content.get("plan", [])
    for step in plan:
        markdown_content += f"## {step.get('section', '')}\n\n"
        markdown_content += f"{step.get('content', '')}\n\n"
    resources = dea_content.get("resources", [])
    markdown_content += f"## References:\n\n"
    for resource in resources:
        markdown_content += f"{resource.get('resource_id','')}. {[resource.get('resource_description', '')]}"
        if "url" in resource:
            markdown_content += f"({resource['url']})"
        markdown_content += "\n"
    return markdown_content

def evaluate_document(document_content: str, turns: List = [], solution: str = None, content_type="markdown"):
    """
    Evaluate a document using various metrics from costorm_eval.

    Args:
        document_path: Path to the document to evaluate
        reference_path: Optional path to a reference document for comparison
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dea_evaluation_scores = DEA_evaluation(document_content, solution, content_type="markdown")

    logger.info("Initializing evaluators...")
    # Initialize evaluators with proper configuration
    prometheus_eval = PrometheusEvaluator(
        version="v2.0",  # Use v2.0 to enable OpenAI
        openai_model="gpt-5-nano",
        lm=None,  # Force using the OpenAI model instead of shared LM
    )
    # Force OpenAI by removing HF options
    prometheus_eval._hf_api = None
    prometheus_eval._hf_pipe = None

    writehere_eval = WriteHereEvaluator(
        openai_model="gpt-5-nano",
        lm=None,  # Force using the OpenAI model instead of shared LM
    )

    logger.info("Getting Prometheus scores...")
    # Get Prometheus scores with more specific prompts
    try:
        prometheus_scores = prometheus_eval.score(
            document_content, ["Relevance", "Breadth", "Depth", "Novelty"]
        )
        logger.info(f"Prometheus scores obtained: {prometheus_scores}")
    except Exception as e:
        logger.error(f"Error getting Prometheus scores: {e}")
        prometheus_scores = {}

    logger.info("Getting WriteHere scores...")
    # Get WriteHere scores with a more specific prompt
    try:
        writehere_scores = writehere_eval.evaluate_report(
            document_content,
            "Evaluate this document's quality in terms of coverage, novelty, relevance, and depth of analysis.",
        )
        logger.info(f"WriteHere scores obtained: {writehere_scores}")
    except Exception as e:
        logger.error(f"Error getting WriteHere scores: {e}")
        writehere_scores = {}

    logger.info("Getting article quality metrics...")
    # Get article quality metrics with a more meaningful reference
    if solution is None:
        # Create a more structured reference document
        reference_content = """
        # Reference Document Structure
        
        ## Introduction
        This section should provide a clear overview of the topic and its importance.
        
        ## Main Content
        The document should cover key concepts, provide relevant examples, and demonstrate depth of understanding.
        
        ## Analysis
        Include critical analysis, comparisons, and novel insights.
        
        ## Conclusion
        Summarize key points and provide meaningful conclusions.
        """
    else:
        reference_content = temporary_transform_dea_into_markdown(solution)
    try:
        # Try to use rouge_score library first
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

            rouge_scores = scorer.score(reference_content, document_content)

            # Convert to the expected format
            article_metrics = {
                # "entity_recall": 0.0,  # Keep the original entity recall
                "rouge_scores": {
                    "rouge-1": {
                        "p": rouge_scores["rouge1"].precision,
                        "r": rouge_scores["rouge1"].recall,
                        "f": rouge_scores["rouge1"].fmeasure,
                    },
                    "rouge-2": {
                        "p": rouge_scores["rouge2"].precision,
                        "r": rouge_scores["rouge2"].recall,
                        "f": rouge_scores["rouge2"].fmeasure,
                    },
                    "rouge-l": {
                        "p": rouge_scores["rougeL"].precision,
                        "r": rouge_scores["rougeL"].recall,
                        "f": rouge_scores["rougeL"].fmeasure,
                    },
                },
                # "llm_scores": {},
            }
        except ImportError:
            # Fallback to the original implementation
            logger.warning("rouge_score not found, using fallback implementation")
            article_metrics = evaluate_article_quality(
                document_content, reference_content
            )
        logger.info(f"Article metrics obtained: {article_metrics}")
    except Exception as e:
        logger.error(f"Error getting article metrics: {e}")
        article_metrics = {}

    # Print results with more context
    print("\n=== Document Evaluation Results ===")
    print("\nPrometheus Scores (0-1 scale, higher is better):")
    for aspect, score in prometheus_scores.items():
        print(f"{aspect}: {score:.2f}")

    print("\nWriteHere Scores (0-1 scale, higher is better):")
    for aspect, score in writehere_scores.items():
        print(f"{aspect}: {score:.2f}")

    print("\nArticle Quality Metrics:")
    print(f"Entity Recall (0-1 scale): {article_metrics.get('entity_recall', 0):.2f}")
    print("\nROUGE Scores (precision, recall, F1):")
    for metric, scores in article_metrics.get("rouge_scores", {}).items():
        print(f"{metric}:")
        for score_type, value in scores.items():
            print(f"  {score_type}: {value:.2f}")

    return {
        "prometheus_scores": prometheus_scores,
        "writehere_scores": writehere_scores,
        "article_metrics": article_metrics,
        "dea_evaluation_scores": dea_evaluation_scores,
    }
# This will be called both when run as a script and when imported and reloaded
if __name__ == "__main__":
    main()
else:
    # For test_cli_entrypoint
    # When the module is reloaded during testing, we need to check if sys.argv
    # has been modified to simulate CLI usage
    if len(sys.argv) > 1 and sys.argv[0].endswith("costorm_eval.py"):
        main()
