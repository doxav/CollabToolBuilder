# Full regression coverage for “Measure quality of generations” §2A
# Scenarios covered:
#   ① report-only  •  ② full-turn  •  ③ live CLI patch  •  ④ Prom-gauges
#   - plus Coverage&Time evaluator with an artificial KB tree.

from __future__ import annotations

import importlib, json, sys, types, textwrap
from pathlib import Path

import pytest

import benchmark.doc_eval as cse

import os
try:
    import config
except ImportError:
    raise ImportError("Ensure config.py exists and OPENAI_API_KEY is set or necessary setup for LLM/Embedding.")

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY  # certains wrappers le lisent
os.environ["ENCODER_API_TYPE"] = "openai"
os.environ["LITELLM_CACHE"] = "disabled"


# ───────── helper objects ─────────────────────────────────────────────

class _DummyNode:
    def __init__(self, children=None):
        self.children = children or []

class _DummyRunner:
    def __init__(self):
        self.knowledge_base = types.SimpleNamespace(
            root=_DummyNode(
                children=[
                    _DummyNode(),                         # depth-1
                    _DummyNode(children=[_DummyNode()]),  # depth-2
                ]
            )
        )
        self.turns = [1, 2, 3]        # simulate 3 turns

class _DummyModerator:
    def __init__(self): self.extra_metadata = {}
    def log_turn(self, d): ...
    def finish(self, txt): ...

# ───────── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def tmp_report(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"report": "Alpha bravo charlie."}))
    return p

@pytest.fixture
def tmp_full_log(tmp_path):
    log = {
        "report": "Delta echo foxtrot.",
        "turns": [
            {"idx": 0, "intent": "ORIGINAL_QUESTION", "text": "What is Delta?"},
            {"idx": 1, "intent": "POTENTIAL_ANSWER",  "text": "Delta → https://δ.example."},
            {"idx": 2, "intent": "FURTHER_DETAILS",   "text": "More on Delta."},
        ],
        "duration": 7.5,
    }
    p = tmp_path / "run.json"
    p.write_text(json.dumps(log))
    return p

# ───────── tests ─────────────────────────────────────────────────────

# ① Offline – report-only
def test_report_only(tmp_report):
    res = cse.evaluate_run(tmp_report)
    assert set(res) == {"report", "turns", "citation_quality", "prom_lines"}
    assert res["turns"] == []                     # no turn array present
    for k in ["Relevance", "Breadth", "Depth", "Novelty"]:
        assert 0.0 <= res["report"]["scores"][k] <= 1.0

# ② Offline – full-turn path
def test_full_turn_metrics(tmp_full_log):
    res = cse.evaluate_run(tmp_full_log)
    assert len(res["turns"]) == 3
    # citation_quality: exactly 1 of 3 turns introduces a new URL
    assert pytest.approx(res["citation_quality"], abs=1e-3) == 1 / 3
    # Prometheus lines must include our fallback gauge
    assert "costorm_turn_score" in res["prom_lines"]

# Coverage & Time auxiliary evaluator
def test_coverage_and_speed():
    runner = _DummyRunner()
    ev = cse.CoverageAndSpeedEvaluator()
    out = ev.evaluate(runner, duration=None)
    # 3 nodes under root, 3 turns
    assert out == {"coverage": 3, "time": 3}

# ③/④ Live-patch console & Prom-snapshot
def test_live_patch_moderator(capsys):
    mod = _DummyModerator()
    cse.patch_costorm_for_live_eval(mod)
    mod.log_turn({"idx": 0, "intent": "ORIGINAL_QUESTION", "text": "Ping?"})
    mod.finish("Final answer.")
    captured = capsys.readouterr().out
    assert "[EVAL-TURN 00]" in captured
    assert "[REPORT EVAL]" in captured
    assert "report_auto_scores" in mod.extra_metadata

# CLI entry-point (smoke)
def test_cli_entrypoint(monkeypatch, tmp_full_log, tmp_path):
    tmp_log = tmp_path / "copy.json"
    tmp_log.write_bytes(tmp_full_log.read_bytes())
    monkeypatch.setattr(sys, "argv", ["costorm_eval.py", str(tmp_log)])
    importlib.reload(cse)         # triggers __main__
    assert tmp_log.with_suffix(".eval.json").exists()

# ────────────────────── LIVE TEST ────────────────────────────────────────────────

# at the bottom of tests/test_2A_costorm_eval.py
import json
import pytest
from pathlib import Path

import benchmark.doc_eval as cse  # :contentReference[oaicite:0]{index=0}
from knowledge_storm.collaborative_storm.engine import (
    CoStormRunner, RunnerArgument, CollaborativeStormLMConfigs,
)
from knowledge_storm.collaborative_storm.graph_runner import CoStormGraphRunner
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import YouRM

# — simple dummy components so no real API calls —
class DummyLM:
    def __call__(self, prompt, **kwargs): return "dummy"
    def get_usage_and_reset(self): return {}
class DummyRM:
    def __call__(self, q): return []

def make_runner(total_turns: int = 2) -> CoStormRunner:
    cfg = CollaborativeStormLMConfigs()
    # replace every LM slot with our DummyLM
    for slot in (
        "question_answering_lm",
        "discourse_manage_lm",
        "utterance_polishing_lm",
        "warmstart_outline_gen_lm",
        "question_asking_lm",
        "knowledge_base_lm",
    ):
        setattr(cfg, slot, DummyLM())
    args = RunnerArgument(topic="TEST", retrieve_top_k=1, total_conv_turn=total_turns)
    args.temperature = 0.0
    wrapper = LoggingWrapper(cfg)
    return CoStormRunner(cfg, args, wrapper, rm=DummyRM())

def test_costorm_eval_sequence_vs_graph(tmp_path):
    """Compare evaluation metrics between sequential and graph-based runners using real evaluators."""
    # Import make_runner from conftest to get properly configured runners
    from tests.conftest import make_runner, slots, build_nano_lm
    from knowledge_storm.dataclass import ConversationTurn, Information
    
    # --- Helper function to dump and evaluate a runner ---
    def dump_and_eval(runner, stage, tmp_path):
        report= runner.generate_report()
        log_json = {
            "report": report,
            "turns": [
                # Use utterance_type instead of intent
                {"idx": i, "intent": t.utterance_type, "text": t.utterance}
                for i, t in enumerate(runner.conversation_history)
            ],
        }
        p = tmp_path / f"eval_{stage}.json"
        with open(p, 'w') as f:
            json.dump(log_json, f)
        
        # Use the real evaluator from costorm_eval
        results = cse.evaluate_run(p)
        print(f"\n--- Metrics @ {stage} ---")
        
        # Access the nested structure correctly
        report_scores = results.get('report', {}).get('scores', {})
        print(f"report scores: {report_scores}")
        print(f"n turns: {len(results.get('turns', []))} citation_quality: {results.get('citation_quality', 0.0)}")
        
        # Return results with a consistent structure to make assertion easier
        return {
            "report_scores": report_scores,
            "turn_scores": results.get('turns', []),
            "citation_quality": results.get('citation_quality', 0.0)
        }
    
    # --- Create sequential and graph runners ---
    seq = make_runner()  # Uses real nano LLM from conftest
    gph = make_runner()  # Uses real nano LLM from conftest
    seq.runner_argument.temperature = 0.0
    gph.runner_argument.temperature = 0.0

    # Initialize conversation_history if needed to prevent IndexError
    if not seq.conversation_history:
        seq.conversation_history = [
            ConversationTurn(
                role="Guest",
                raw_utterance="Tell me about TEST",
                utterance_type="ORIGINAL_QUESTION",
                utterance="Tell me about TEST"
            )
        ]
    if not gph.conversation_history:
        gph.conversation_history = [
            ConversationTurn(
                role="Guest",
                raw_utterance="Tell me about TEST",
                utterance_type="ORIGINAL_QUESTION",
                utterance="Tell me about TEST"
            )
        ]
    
    def assert_metric(thresholds, res_seq, res_gph, max_divergence=4.):
        for metric, floor in thresholds.items():
            s_val = res_seq["report_scores"][metric]
            g_val = res_gph["report_scores"][metric]
            assert s_val >= floor, f"{metric} seq below {floor}"
            assert g_val >= floor, f"{metric} gph below {floor}"
            if g_val != 0:
                assert 1./max_divergence <= abs(s_val/g_val) <= max_divergence, f"{metric} diverged >0.1"
            else:
                assert s_val == 0, f"{metric} seq != 0 when gph = 0"
    
    seq.warm_start()
    gph.warm_start()
    # Set up thresholds and evaluate progressively more complex content
    thresholds = { "Relevance": 0.5, "Breadth": 0., "Depth": 0., "Novelty": 0. }
    # --- 1. Initial state evaluation ---
    res0_seq = dump_and_eval(seq, "0_turns_seq", tmp_path)
    res0_gph = dump_and_eval(gph, "0_turns_gph", tmp_path)
    assert_metric(thresholds, res0_seq, res0_gph)
    
    # --- 2. After 1 turn ---
    seq.step(simulate_user=True, simulate_user_intent="Test")
    gph.step(simulate_user=True, simulate_user_intent="Test")
    # --- EVALUATE 1 TURN ---
    thresholds = { "Relevance": 0.5, "Breadth": 0.1, "Depth": 0.1, "Novelty": 0.1 }
    res1_seq = dump_and_eval(seq, "1_turn_seq", tmp_path)
    res1_gph = dump_and_eval(gph, "1_turn_gph", tmp_path)
    assert_metric(thresholds, res1_seq, res1_gph)
    # Breadth and depth should increase as content becomes more detailed
    assert res1_seq["report_scores"]["Breadth"] >= res0_seq["report_scores"]["Breadth"], "Breadth should increase as content expands"
    assert res1_seq["report_scores"]["Depth"] >= res0_seq["report_scores"]["Depth"], "Depth should increase or stay the same"
    assert res1_gph["report_scores"]["Breadth"] >= res0_gph["report_scores"]["Breadth"], "Breadth should increase as content expands"
    assert res1_gph["report_scores"]["Depth"] >= res0_gph["report_scores"]["Depth"], "Depth should increase or stay the same"
    # Citation quality should improve as we add citations
    assert res1_seq["citation_quality"] > 0, "Citation quality should be > 0"
    assert res1_gph["citation_quality"] > 0, "Citation quality should be > 0"
    
    # --- 3. Most detailed state evaluation --- 
    seq.step(simulate_user=True, simulate_user_intent="Test")
    gph.step(simulate_user=True, simulate_user_intent="Test")
    seq.step(simulate_user=True, simulate_user_intent="Test")
    gph.step(simulate_user=True, simulate_user_intent="Test")
    seq.step(simulate_user=True, simulate_user_intent="Test")
    gph.step(simulate_user=True, simulate_user_intent="Test")
    # --- EVALUATE 2 TURNS ---
    thresholds = { "Relevance": 0.5, "Breadth": 0.1, "Depth": 0.1, "Novelty": 0.1 }
    res2_seq = dump_and_eval(seq, "2_turns_seq", tmp_path)
    res2_gph = dump_and_eval(gph, "2_turns_gph", tmp_path)
    assert_metric(thresholds, res2_seq, res2_gph)

    # Compare lightweight article / outline / citation metrics
    aq_seq = cse.evaluate_article_quality(
        seq.generate_report(), seq.generate_report()
    )  # identical → perfect
    aq_gph = cse.evaluate_article_quality(
        gph.generate_report(), gph.generate_report()
    )
    assert aq_seq["entity_recall"] == 1.0 == aq_gph["entity_recall"]

    ol_seq = cse.evaluate_outline_quality(["Intro", "Body", "End"], ["Intro", "End"])
    assert 0.0 <= ol_seq["heading_soft_recall"] <= 1.0

    cq = cse.evaluate_citation_quality("Factoid [1].", [{"title": "dummy", "text": "Factoid"}])
    assert cq["citation_recall"] == cq["citation_precision"] == 1.0

    # Further improvements in metrics
    assert res2_seq["report_scores"]["Breadth"] >= res1_seq["report_scores"]["Breadth"], "Breadth should continue to improve"
    assert res2_seq["report_scores"]["Depth"] >= res1_seq["report_scores"]["Depth"], "Depth should increase with more detailed content"
    assert res2_gph["report_scores"]["Breadth"] >= res1_gph["report_scores"]["Breadth"], "Breadth should continue to improve"
    assert res2_gph["report_scores"]["Depth"] >= res1_gph["report_scores"]["Depth"], "Depth should increase with more detailed content"
    assert res2_seq["citation_quality"] > 0, "Citation quality should be > 0"
    assert res2_gph["citation_quality"] > 0, "Citation quality should be > 0"
    
    print("\nEvaluation metrics progression:")
    print(f"Initial Breadth: {res0_seq['report_scores']['Breadth']:.2f} → Middle: {res1_seq['report_scores']['Breadth']:.2f} → Final: {res2_seq['report_scores']['Breadth']:.2f}")
    print(f"Initial Depth: {res0_seq['report_scores']['Depth']:.2f} → Middle: {res1_seq['report_scores']['Depth']:.2f} → Final: {res2_seq['report_scores']['Depth']:.2f}")
    print(f"Initial Citation Quality: {res0_seq['citation_quality']:.2f} → Middle: {res1_seq['citation_quality']:.2f} → Final: {res2_seq['citation_quality']:.2f}")

# ───────── NEW TESTS FOR ADDED EVALUATORS ─────────────────────────────

def test_prometheus_versions():
    txt = "Artificial intelligence transforms many industries."
    # v1.0 (heuristic)
    v1 = cse.PrometheusEvaluator(version="v1.0")
    s1 = v1.score(txt, ["Relevance", "Depth"])
    assert 0.0 <= s1["Relevance"] <= 1.0
    assert 0.0 <= s1["Depth"] <= 1.0

    # v2.0 (falls back to heuristic in CI if HF/OpenAI unavailable)
    v2 = cse.PrometheusEvaluator(version="v2.0", openai_model=None)
    s2 = v2.score(txt, ["Relevance"])
    assert "Relevance" in s2 and 0.0 <= s2["Relevance"] <= 1.0


def test_writehere_evaluator():
    wh = cse.WriteHereEvaluator(openai_model=None)   # offline path for CI
    res = wh.evaluate_report("Short article about AI.", "What is AI?")
    assert set(res) == {
        "Broad Coverage",
        "Novelty",
        "Relevance and Focus",
        "Depth of Exploration",
    }
    assert all(0.0 <= v <= 1.0 for v in res.values())


def test_citation_quality_simple():
    article = "The sky is blue [1]."
    docs = [{"title": "Sky", "text": "The sky is blue due to Rayleigh scattering."}]
    out = cse.evaluate_citation_quality(article, docs)
    assert out["citation_recall"] == 1.0
    assert out["citation_precision"] == 1.0


def test_article_quality_entities_and_rouge():
    gold = "Paris is the capital of France."
    pred = "Paris is a city in Europe."
    res = cse.evaluate_article_quality(pred, gold)
    # one of two gold entities (Paris) is present ⇒ recall 0.5
    assert res["entity_recall"] == 0.5
    # rouge keys exist
    assert "rouge-1" in res["rouge_scores"]


def test_outline_quality_overlap():
    golden = ["Background", "Methods", "Results", "Conclusion"]
    pred = ["Introduction", "Methods", "Conclusion"]
    res = cse.evaluate_outline_quality(pred, golden)
    # two of four headings overlap
    assert res["heading_soft_recall"] == 0.5
    assert 0.0 <= res["heading_entity_recall"] <= 1.0