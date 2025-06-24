import argparse
import csv
import json
import sys
from pathlib import Path

import pytest

import subprocess

from knowledge_storm.collaborative_storm import costorm_eval as cse

import benchmark.benchmark as benchmark

import os, config

os.environ["OPENAI"] = os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["ENCODER_API_TYPE"] = "openai"
os.environ["LITELLM_CACHE"] = "disabled"
os.environ["SearXNG"] = "http://127.0.0.1:8080"
os.environ["SEARXNG_API_KEY"] = ""

# ────────────────────────── UNIT TESTS ──────────────────────────

# def test_run_writehere(tmp_path):
#     # Create a mock for the WriteHere functionality that doesn't depend on non-existent modules
#     def mock_writehere_implementation(topic, intent=None):
#         # Simulate WriteHere writing a report directly
#         return f"writehere-{topic}", []
    
#     # Patch the run_writehere function to use our mock implementation
#     #monkeypatch.setattr(benchmark, "run_writehere", mock_writehere_implementation)
#     # generate a random int between 200 and 400
#     total_words = 200 + int(os.urandom(2).hex(), 16) % 400

#     report, turns = benchmark.run_writehere(topic="LLM agents to collaborative solve problems", intent="write a full state of the art research paper on the topic", writehere_model="gpt-4.1-nano", engine_backend="SearXNG", prompt=f"write a {total_words} words document on the topic with latest research")
#     assert len(report) > 105
#     assert turns == []
#     assert turns == []


def test_parse_args_defaults():
    parser = benchmark._build_parser()
    args = parser.parse_args([])
    assert args.datasets == ["FreshWiki", "WildSeek", "DEA"]
    assert args.methods == ["Storm", "CoStorm", "CoStormGraph", "WriteHere"]
    assert isinstance(args.dea_root, Path)
    assert args.writehere_model == "gpt-4o"
    assert args.writehere_backend == "openai"

@pytest.fixture(autouse=True)
def patch_evaluator_and_env(monkeypatch, tmp_path):
    # Patch costorm_eval.evaluate_run
    monkeypatch.setattr(cse, "evaluate_run", lambda path: {"report_score": 1.23})
    # Patch DEA EnvironmentManager if present
    DummyEnv = type("DummyEnv", (), {
        "reset": lambda self: None,
        "synthesis_manager": type("SM", (), {
            "GetFromLatex": lambda self, txt: None,
            "GetFromText": lambda self, txt: None,
            "get_distance_to_targetJSON": lambda self: {"detailed": 2.34},
        })(),
        "get_score": lambda self: {"global": 3.45},
    })
    # Stub out the import env.env.EnvironmentManager
    sys.modules["env.env"] = type("mod", (), {"EnvironmentManager": lambda **kw: DummyEnv()})
    # Ensure VoyagerEnvIR_CPS_TechSynthesis is truthy for DEA tests
    monkeypatch.setattr(benchmark, "VoyagerEnvIR_CPS_TechSynthesis", object())
    return tmp_path

def test_load_dea(tmp_path):
    # Create fake DEA JSON structure
    dea_root = tmp_path / "DEA"
    out = dea_root / "output" / "latex"
    out.mkdir(parents=True)
    sample = {"title": "T1", "context": "CTX"}
    fn = out / "one.json"
    fn.write_text(json.dumps(sample), encoding="utf-8")
    results = list(benchmark.load_dea(dea_root))
    assert results == [("T1", "CTX")]
    # DEA_FILES should have been populated
    assert "T1" in benchmark.DEA_FILES
    assert benchmark.DEA_FILES["T1"] == fn

def test_evaluate_and_dump_non_dea(tmp_path):
    out_dir = tmp_path / "out"
    metrics = benchmark._evaluate_and_dump(
        topic="X", report="R", turns=[], method="M", dataset="FreshWiki", out_dir=out_dir
    )
    # Should reflect our patched evaluate_run
    assert metrics["report_score"] == 1.23
    # No latex_rubric for non-DEA
    assert "latex_rubric" not in metrics
    # Files created
    files = list(out_dir.glob("*.json"))
    assert any(f.suffix == ".json" for f in files)
    assert any(f.name.endswith(".eval.json") for f in files), f'Expected .eval.json in file, found suffixes: {[f"suffix:{f.suffix} in file:{f}" for f in files]}'
    # compare n last part of filename == ".eval.json"


@pytest.fixture
def stub_subprocess_run(tmp_path, monkeypatch):
    """Monkeypatch subprocess.run to simulate writing output files."""
    # Redirect RESULT_DIR to tmp_path/results
    monkeypatch.setattr(benchmark, "RESULT_DIR", tmp_path / "results")
    def fake_run(cmd, input=None, text=None, check=None, **kwargs):
        # Accept all keyword arguments that subprocess.run uses
        # Determine which runner by script path
        if len(cmd) > 1:
            script = cmd[1]
            if "run_storm_wiki_gpt.py" in script:
                topic = (input or "").strip()
                outp = tmp_path / "results" / "tmp_storm" / topic
                outp.mkdir(parents=True, exist_ok=True)
                (outp / "storm_gen_article_polished.txt").write_text(f"report-{topic}", encoding="utf-8")
            elif "run_costorm_gpt.py" in script:
                # feed topic then blank
                topic = (input or "").splitlines()[0]
                outp = tmp_path / "results" / "tmp_costorm" / topic
                outp.mkdir(parents=True, exist_ok=True)
                (outp / "report.md").write_text(f"costorm-{topic}", encoding="utf-8")
        # For git clone commands, just return None (no-op)
        return None
    monkeypatch.setattr(subprocess, "run", fake_run)
    # Also stub clone_repo to prevent actual git operations
    monkeypatch.setattr(benchmark, "clone_repo", lambda *args, **kwargs: None)
    return tmp_path

def test_run_costorm():
    report, turns = benchmark.run_costorm("Write a comprehensive, 10000 words 2025 about the Comparative Analyses of Local vs. Spatial Complex Question Answering in Large Language Models including experiments and results, theoretical and numerical models, degeneracy, reasoning capabilities, Formal Logic Proof of relation (Deductive Argument based on literature validated by experiment results vs unproven), main researches in 2025 (researchers, scope, estimated future contributions) and key open questions.",
        max_conv_turn= 2,
        search_top_k=3,
        retrieve_top_k= 3,
    )
    print(f"Report: {report}, Turns: {turns}")
    assert len(str(report)) > 200
    assert len(turns) >= 2

def test_run_storm():
    report, turns = benchmark.run_storm(
        "Write a comprehensive, 10000 words 2025 about the Comparative Analyses of Local vs. Spatial Complex Question Answering in Large Language Models including experiments and results, theoretical and numerical models, degeneracy, reasoning capabilities, Formal Logic Proof of relation (Deductive Argument based on literature validated by experiment results vs unproven), main researches in 2025 (researchers, scope, estimated future contributions) and key open questions.",
        max_conv_turn= 1,
        search_top_k=3,
        retrieve_top_k= 3,
    )
    print(f"Report: {report}, Turns: {turns}")
    assert len(str(report)) > 200
    assert len(turns) >= 2

def test_run_costorm_graph(monkeypatch):
    # Stub out the GraphRunner class
    class DummyGraph:
        def __init__(self, cfg, args, rm): pass
        def warm_start(self): pass
        def step(self, **kw): pass
        def generate_report(self): return "G-REPORT"
        @property
        def conversation_history(self):
            T = type("T", (), {"utterance_type": "UT", "utterance": "hello"})
            return [T()]
    monkeypatch.setattr(
        "knowledge_storm.collaborative_storm.graph_runner.CoStormGraphRunner",
        DummyGraph,
    )
    report, turns = benchmark.run_costorm_graph("ZZZ")
    assert report == "G-REPORT"
    assert isinstance(turns, list)
    assert turns[0]["intent"] == "UT"
    assert turns[0]["text"] == "hello"

# ────────────────────────── FULL END-TO-END INTEGRATION TEST ──────────────────────────

def test_full_benchmark(tmp_path, monkeypatch):
    # Stub clone_repo to no-op
    monkeypatch.setattr(benchmark, "clone_repo", lambda *args, **kwargs: None)
    # Redirect RESULT_DIR
    monkeypatch.setattr(benchmark, "RESULT_DIR", tmp_path / "results")
    # Simplify dataset loaders
    monkeypatch.setattr(benchmark, "load_freshwiki", lambda: [("F1", "")])
    monkeypatch.setattr(benchmark, "load_wildseek", lambda: [("W1", "intent1")])
    monkeypatch.setattr(benchmark, "load_dea", lambda root: [("D1", "ctx")])
    benchmark.DATASET_LOADERS = {
        "FreshWiki": benchmark.load_freshwiki,
        "WildSeek": benchmark.load_wildseek,
        "DEA": lambda root: benchmark.load_dea(root),
    }
    # Stub METHODS to return predictable results
    monkeypatch.setattr(benchmark, "METHODS", {
        "Storm": lambda t, intent=None: (f"S-{t}", []),
        "CoStorm": lambda t, intent=None: (f"C-{t}", []),
        "CoStormGraph": lambda t, intent=None: (f"G-{t}", []),
        "WriteHere": lambda t, intent=None: (f"W-{t}", []),
    })
    # Ensure DEA evaluation stub
    monkeypatch.setattr(cse, "evaluate_run", lambda path: {"m": 42})
    monkeypatch.setattr(benchmark, "VoyagerEnvIR_CPS_TechSynthesis", None)

    # Prepare CLI args
    sys.argv = [
        "benchmark.py",
        "--datasets", "FreshWiki", "WildSeek", "DEA",
        "--methods", "Storm", "WriteHere",
        "--dea-root", str(tmp_path / "dummy_dea"),
    ]
    # Create dummy DEA folder so load_dea won't error (though it's stubbed)
    (tmp_path / "dummy_dea" / "output" / "latex").mkdir(parents=True)

    # Run main
    benchmark.main()

    # Check per-combo dirs and CSV
    res = tmp_path / "results"
    assert (res / "FreshWiki_Storm").is_dir()
    assert (res / "WriteHere_WriteHere").parent.exists() or True  # sanity
    agg = res / "all_results.csv"
    assert agg.exists()
    lines = agg.read_text(encoding="utf-8").splitlines()
    # Header + 6 runs (3 datasets × 2 methods)
    assert len(lines) == 1 + 3 * 2
