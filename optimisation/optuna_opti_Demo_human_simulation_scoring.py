import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo import multi_agent_research_generation_persist_at_the_end, score_generated_report_with_existing_env

TOPICS = [
    ("Complex QA and language models hybrid architectures, Survey", "..."),
    ("Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature", "..."),
]

MODES = {
    "baseline":   {"mode": "intelligent",    "analysts": 1, "iters": 1},
    "critique":   {"mode": "critique",       "analysts": 1, "iters": 2},
    "multi_exp":  {"mode": "multi_expert",   "analysts": 3, "iters": 1},
}

def run_all():
    results = {}
    for name, cfg in MODES.items():
        results[name] = {}
        for title, topic in TOPICS:
            print(f"\n=== Running {name} on: {title[:40]} ===")
            latex = multi_agent_research_generation_persist_at_the_end(
                title=title,
                topic=topic,
                max_analysts=cfg["analysts"],
                max_report_iterations=cfg["iters"],
                automation="full_auto",
                human_simulation_mode=cfg["mode"],
            )
            score_dict = score_generated_report_with_existing_env(
                final_report_in_latex=latex,
                topic=topic,
                title=title,
            )
            score = score_dict.get("scores", -1)
            print(f"==> SCORE = {score}")
            results[name][title] = score
    return results

if __name__ == "__main__":
    run_all()