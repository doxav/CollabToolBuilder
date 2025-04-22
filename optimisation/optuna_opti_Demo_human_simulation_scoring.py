"""
Optuna search – Demo: impact of the *timing* and *mode* of human simulation
===============================================================================

The search space:
    - sim_mode ∈ {"intelligent", "critique", "multi_expert"}
    - exp_name  ∈ {
          "A_planner", "A_writer", "A_critic", "A_analyst",
          "B_plan_write", "B_plan_critic",
          "C_critic×2",
          "D_plan→critic"
      }

Each trial = (sim_mode, exp_name) executed on the **2 predefined subjects**.  
Returned score: average of the two scores; we *maximize* the Optuna score.  
All runs—including those that crash—are logged for offline analysis.

Local execution:
"""

import os, sys, json, traceback, statistics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo import multi_agent_research_generation_persist_at_the_end, score_generated_report_with_existing_env
from optimisation.optuna_main import global_main      # ⬅️ like the other optimization scripts
import hashlib

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
    #"A_writer"    : dict(iters=1, agents=("section_writer",)),
    "A_critic"    : dict(iters=1, agents=("critic",)),
    "A_analyst"   : dict(iters=1, agents=("analyst",)),
    # B. 1 it / 2 agents
    #"B_plan_write": dict(iters=1, agents=("planner", "section_writer")),
    "B_plan_critic":dict(iters=1, agents=("planner", "critic")),
    "B_analyst_plan":dict(iters=1, agents=("planner", "analyst")),
    "B_analyst_critic":dict(iters=1, agents=("analyst", "critic")),
    # C. 2 it / same agent
    "C_critic×2"  : dict(iters=2, agents=("critic",)),
    "C_plan×2"    : dict(iters=2, agents=("planner",)),
    "C_analyst×2" : dict(iters=2, agents=("analyst",)),
    # D. 2 it / sequence
    "D_plan→analyst":dict(iters=2, agents=("planner", "analyst")),
    "D_plan→critic":dict(iters=2, agents=("planner", "critic")),
    "D_analyst→critic":dict(iters=2, agents=("analyst", "critic")),
}
# --------------------------------------------------------------------------- #
def safe_run(title: str, topic: str, sim_mode: str, exp_cfg: dict, name_xp: str, trial, exp_name: str):
    """
    Executes a trial → returns (average_score, report_basename, scores_detail)
    - Keeps ALL detailed scores (plan_embedding_similarity, …)
    - Saves complete .tex + .json files
    - Never raises unhandled exceptions (returns -999)
    """
    report_basename = "N/A"
    try:
        # ---------- generation ----------
        latex = multi_agent_research_generation_persist_at_the_end(
            title=title,
            topic=topic,
            max_analysts=3,
            max_report_iterations=exp_cfg["iters"],
            automation="full_auto",
            human_simulation_mode=sim_mode,
            agents_with_human=exp_cfg["agents"],
        )

        # ---------- hashing + dossier ----------
        hash_id = hashlib.md5(latex.encode()).hexdigest()[:8]
        report_basename = f"{sim_mode}_{exp_cfg['agents']}_{title[:10].replace(' ', '_')}_{hash_id}"
        report_dir = os.path.join("Optuna_report_outputs", name_xp)
        os.makedirs(report_dir, exist_ok=True)

        # ---------- save .tex ----------
        tex_path = os.path.join(report_dir, f"{report_basename}.tex")
        with open(tex_path, "w") as f:
            f.write(latex)

        # ---------- scoring ----------
        scores_dict = score_generated_report_with_existing_env(latex, topic, title)
        scores_detail = scores_dict.get("scores", {}) or {}          # dict complet
        # métrique principale = moyenne simple des valeurs numériques présentes
        numeric_vals = [v for v in scores_detail.values() if isinstance(v, (int, float)) and v != -1]
        sc_main = statistics.fmean(numeric_vals) if numeric_vals else -1.0

        # ---------- save .json complet ----------
        json_path = os.path.join(report_dir, f"{report_basename}.json")
        with open(json_path, "w") as f:
            json.dump({
                "trial": trial.number,
                "sim_mode": sim_mode,
                "exp_name": exp_name,
                "agents": list(exp_cfg["agents"]),
                "topic": topic,
                "title": title,
                "score_main": sc_main,
                "scores_all": scores_detail,
                "qualitative": scores_dict.get("qualitative feedback", ""),
                "report_file": os.path.basename(tex_path),
            }, f, indent=2)

        return sc_main, report_basename, scores_detail

    except Exception as e:
        print(f"⚠️ Crash on {title[:25]} | {sim_mode} | {exp_cfg} -> {e}")
        traceback.print_exc()
        # même structure mais vide
        return -999.0, report_basename, {}

# ---------------------------  Optuna objective  ---------------------------- #
def objective(trial, name_xp: str):
    sim_mode  = trial.suggest_categorical("sim_mode", SIM_MODES)
    exp_name  = trial.suggest_categorical("exp_name", list(EXPERIMENTS.keys()))
    exp_cfg   = EXPERIMENTS[exp_name]

    scores_subjects = []       # one value per subject
    scores_detail_all = {}     # will concatenate the two detailed dictionaries

    for title, topic in TOPICS:
        s, report_basename, detail = safe_run(
            title, topic, sim_mode, exp_cfg, name_xp, trial, exp_name
        )
        scores_subjects.append(s)
        scores_detail_all[f"{title[:15]}"] = detail   # shorten key

    avg_score = statistics.fmean(scores_subjects) if scores_subjects else -999.0

    # ---- Persist summary & path to details ----
    os.makedirs("Optuna_results", exist_ok=True)
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(json.dumps({
            "trial": trial.number,
            "sim_mode": sim_mode,
            "exp_name": exp_name,
            "scores": scores_subjects,
            "avg": avg_score,
            "detail_file": report_basename + ".json"
        }) + "\n")

    # ---- Attributs Optuna pour visualisation ----
    trial.set_user_attr("sim_mode", sim_mode)
    trial.set_user_attr("exp_name",  exp_name)
    trial.set_user_attr("score_subjects", scores_subjects)
    trial.set_user_attr("score_details", scores_detail_all) 

    return avg_score

# ---------------------------  CLI / launcher  ------------------------------ #
if __name__ == "__main__":
    # argv1 : name_exp ; argv2 : n_trials
    import datetime, os
    default_name = f"demo_human_sim_{datetime.datetime.now():%Y%m%d_%H%M}"
    name_exp = sys.argv[1] if len(sys.argv) > 1 else default_name
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    global_main(objective, "demo", "human_sim", name_exp=name_exp, n_trials=n_trials)
