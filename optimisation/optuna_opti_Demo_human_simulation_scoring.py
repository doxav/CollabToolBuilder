"""
Optuna search – Demo : impact du *moment* et du *mode* de la simulation humaine
===============================================================================

Le search space :
    - sim_mode ∈ {"intelligent", "critique", "multi_expert"}
    - exp_name  ∈ {
          "A_planner", "A_writer", "A_critic", "A_analyst",
          "B_plan_write", "B_plan_critic",
          "C_critic×2",
          "D_plan→critic"
      }

Chaque trial = (sim_mode, exp_name) exécuté sur les **2 sujets** imposés.  
Score retourné : moyenne des deux scores ; on *maximise* le score Optuna.  
Tous les runs—y compris ceux qui crasheraient—sont logués pour analyse offline.

Exécution en local :
```bash
python optimisation/optuna_opti_Demo_human_simulation_scoring.py my_xp_name 60
#            └──────── name_exp          └── n_trials (défaut : 100)
```
"""

import os, sys, json, traceback, statistics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo import multi_agent_research_generation_persist_at_the_end, score_generated_report_with_existing_env
from optimisation.optuna_main import global_main      # ⬅️ comme les autres opti-scripts citeturn5file18

# --------------------------------------------------------------------------- #
TOPICS = [
    ("Complex QA and language models hybrid architectures, Survey",
     "Complex QA and language models hybrid architectures, Survey"),
    ("Macroeconomic Effects of Inflation Targeting A Survey of the Empirical Literature",
     "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical Literature"),
]

SIM_MODES = ["intelligent", "critique", "multi_expert"]

EXPERIMENTS = {
    # A. 1 it / 1 agent
    "A_planner"   : dict(iters=1, agents=("planner",)),
    "A_writer"    : dict(iters=1, agents=("section_writer",)),
    "A_critic"    : dict(iters=1, agents=("critic",)),
    "A_analyst"   : dict(iters=1, agents=("analyst",)),
    # B. 1 it / 2 agents
    "B_plan_write": dict(iters=1, agents=("planner", "section_writer")),
    "B_plan_critic":dict(iters=1, agents=("planner", "critic")),
    # C. 2 it / same agent
    "C_critic×2"  : dict(iters=2, agents=("critic",)),
    # D. 2 it / sequence
    "D_plan→critic":dict(iters=2, agents=("planner", "critic")),
}
# --------------------------------------------------------------------------- #

def safe_run(title: str, topic: str, sim_mode: str, exp_cfg: dict):
    """Renvoie un score float ou -999 si crash, + trace dans le log json."""
    try:
        latex = multi_agent_research_generation_persist_at_the_end(
            title=title,
            topic=topic,
            max_analysts=3,
            max_report_iterations=exp_cfg["iters"],
            automation="full_auto",
            human_simulation_mode=sim_mode,
            agents_with_human=exp_cfg["agents"],
        )
        sc = score_generated_report_with_existing_env(latex, topic, title).get("scores", -1)
        # score peut être un dict → on prend la première valeur num.
        if isinstance(sc, dict):
            sc = next((v for v in sc.values() if isinstance(v, (int, float))), -1)
        return float(sc)
    except Exception as e:
        print(f"⚠️ Crash on {title[:25]} | {sim_mode} | {exp_cfg} -> {e}")
        traceback.print_exc()
        return -999.0

# ---------------------------  Optuna objective  ---------------------------- #
def objective(trial, name_xp: str):
    sim_mode  = trial.suggest_categorical("sim_mode", SIM_MODES)
    exp_name  = trial.suggest_categorical("exp_name", list(EXPERIMENTS.keys()))
    exp_cfg   = EXPERIMENTS[exp_name]

    scores = []
    for title, topic in TOPICS:
        s = safe_run(title, topic, sim_mode, exp_cfg)
        scores.append(s)

    avg_score = statistics.fmean(scores) if scores else -999.0

    # ---- Persist trial results for post‑mortem ----
    os.makedirs("Optuna_results", exist_ok=True)
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(json.dumps({
            "trial": trial.number,
            "sim_mode": sim_mode,
            "exp_name": exp_name,
            "scores": scores,
            "avg": avg_score,
        }) + "\n")

    # Plus lisible dans Optuna Board
    trial.set_user_attr("sim_mode", sim_mode)
    trial.set_user_attr("exp_name",  exp_name)
    trial.set_user_attr("scores",    scores)

    # On *maximise* le score ; -999 = KO
    return avg_score

# ---------------------------  CLI / launcher  ------------------------------ #
if __name__ == "__main__":
    # argv1 : name_exp ; argv2 : n_trials
    name_exp = sys.argv[1] if len(sys.argv) > 1 else "demo_human_sim"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    global_main(objective, "demo", "human_sim", name_exp=name_exp, n_trials=n_trials)
