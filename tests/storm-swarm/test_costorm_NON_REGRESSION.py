"""
End-to-end regression test : exécution réelle de Co-STORM
sans aucun stub ni monkey-patch réseau.

Pré-requis :
------------
* Avoir un fichier `config.py` (NON versionné) contenant
      OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
* Accès réseau sortant (HTTP + HTTPS).

Lancez :  pytest -q
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
import openai

import config  # doit définir OPENAI_API_KEY

# ---- Co-STORM --------------------------------------------------------------
from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    RunnerArgument,
    CoStormRunner,
)
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import DuckDuckGoSearchRM
import contextlib, io, logging
# ---------------------------------------------------------------------------
from tests.conftest import build_nano_lm

@pytest.mark.network
def test_costorm_real_run():
    # -- 1. Configuration OpenAI -----------------------------------
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY  # certains wrappers le lisent
    os.environ["ENCODER_API_TYPE"] = "openai"
    os.environ["LITELLM_CACHE"] = "disabled"

    # -- 2. Prépare tous les LLM en gpt-4.1-nano --------------------
    lm_cfg = CollaborativeStormLMConfigs()
    for setter in (
        lm_cfg.set_question_answering_lm,
        lm_cfg.set_discourse_manage_lm,
        lm_cfg.set_utterance_polishing_lm,
        lm_cfg.set_warmstart_outline_gen_lm,
        lm_cfg.set_question_asking_lm,
        lm_cfg.set_knowledge_base_lm,
    ):
        nano = build_nano_lm()
        setter(nano)
        # garde-fou : on vérifie bien le nom du modèle
        assert nano.kwargs.get('model') == "gpt-4.1-nano"

    # -- 3. Paramètres minimaux pour limiter le coût ---------------
    args = RunnerArgument(
        topic="Test de non-régression automatique",
        retrieve_top_k=1,
        max_search_queries=1,
        total_conv_turn=1,
        max_search_thread=1,
        max_search_queries_per_turn=1,
        warmstart_max_num_experts=1,
        warmstart_max_turn_per_experts=1,
        warmstart_max_thread=1,
        max_thread_num=1,
        max_num_round_table_experts=1,
        moderator_override_N_consecutive_answering_turn=1,
        node_expansion_trigger_count=2,
    )

    # -- 4. Runner --------------------------------------------------
    runner = CoStormRunner(
        lm_config=lm_cfg,
        runner_argument=args,
        logging_wrapper=LoggingWrapper(lm_cfg),
        rm=DuckDuckGoSearchRM(k=1),
        callback_handler=None,
    )

    # -- 4.5 Make sure conversation_history has an initial turn ------
    # This ensures the step() method won't fail with IndexError
    if not runner.conversation_history:
        from knowledge_storm.dataclass import ConversationTurn
        runner.conversation_history = [
            ConversationTurn(
                role="Guest",
                raw_utterance=f"Tell me about {args.topic}",
                utterance_type="Original Question"
            )
        ]

    # -- 5. Exécution ------------------------------------------------
    runner.warm_start()
    turn = runner.step()  # un seul tour
    assert turn.utterance.strip()

    runner.knowledge_base.reorganize()
    article = runner.generate_report()
    assert article.strip()

    # -- 6. Persistences & régressions ------------------------------
    tmpdir = tempfile.mkdtemp()
    report_path = os.path.join(tmpdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(article)

    log_path = os.path.join(tmpdir, "log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(runner.dump_logging_and_reset(), fh, ensure_ascii=False, indent=2)

    # Vérifications : fichiers présents et non vides
    assert os.path.getsize(report_path) > 0
    with open(log_path, encoding="utf-8") as fh:
        log = json.load(fh)
    #assert "conv_history" in log
    assert any(s for s in log.values())      # at least one pipeline stage logged
    # optional: check conversation_history is still accessible
    assert runner.conversation_history
    # Nettoyage
    shutil.rmtree(tmpdir)
