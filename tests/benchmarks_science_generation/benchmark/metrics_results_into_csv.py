import os
import json
import csv
import re
from datetime import datetime

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionaries into dot-key dict."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_folder(folder_name: str):
    """Extract ID, datetime, and TOPIC from folder name ID_DATE_TOPIC"""
    parts = folder_name.split("_", 2)
    if len(parts) == 3:
        id = parts[0]
        date_str = parts[1]  # e.g. 20250827-103624
        try:
            dt = datetime.strptime(date_str, "%Y%m%d-%H%M%S")
        except ValueError:
            dt = None
        topic = parts[2]
        return id, dt, topic
    return folder_name, None, folder_name

def extract_agent_labels(config_path):
    """Return merged agent labels string with dynamic config + models."""
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    agents_info = []
    no_dyn_models = []

    for cfg in data if isinstance(data, list) else []:
        agent = cfg.get("agent_name", "")
        dyn = cfg.get("dynamic_llm_config", {}) or {}

        # models
        default_model = dyn.get("default_model") or cfg.get("default_model", "")
        premium_model = dyn.get("premium_model") or cfg.get("premium_model", "")
        model_str = f"default={default_model}, premium={premium_model}" if default_model or premium_model else ""

        # dynamic keys (excluding models)
        dyn_keys = [k for k in dyn.keys() if k not in ("default_model", "premium_model")]

        if dyn_keys:
            label = f"{agent}(" + " | ".join(dyn_keys)
            if model_str:
                label += f", {model_str}"
            label += ")"
            agents_info.append(label)
        else:
            no_dyn_models.append((agent, model_str))

    # compress shared models
    model_values = {m for _, m in no_dyn_models if m}
    if len(model_values) == 1:
        shared_model = model_values.pop()
        agents_info.extend([a for a, _ in no_dyn_models])
        agents_info.append(f"({shared_model})")
    else:
        agents_info.extend([f"{a}({m})" if m else a for a, m in no_dyn_models])

    if not agents_info:
        return None
    return " + ".join(agents_info)

def sort_metrics_files(files):
    """Return list of metrics files with metrics.json last, others sorted numerically."""
    numbered = []
    plain = []
    for f in files:
        if f == "metrics.json":
            plain.append(f)
        elif f.startswith("metrics_") and f.endswith(".json"):
            m = re.match(r"metrics_(\d+)\.json", f)
            if m:
                numbered.append((int(m.group(1)), f))
            else:
                numbered.append((999999, f))
    numbered.sort()
    return [f for _, f in numbered] + plain

def collect_metrics(root_dir="./results", output_csv="./results/all_metrics.csv"):
    rows = []
    fieldnames = ["topic", "agent_name", "metrics_file"]

    topic_written = {}
    agent_written = {}

    # collect all folders with human_llm_config.json
    candidate_folders = []
    for subdir, _, files in os.walk(root_dir):
        folder_name = os.path.basename(subdir)
        if "human_llm_config.json" in files and "_" in folder_name:
            id, dt, topic = parse_folder(folder_name)
            candidate_folders.append((dt or datetime.min, id, subdir, topic))

    # sort by datetime then by ID
    candidate_folders.sort(key=lambda x: (x[0], x[1]))

    for _, _, subdir, topic in candidate_folders:
        agent_label = extract_agent_labels(os.path.join(subdir, "human_llm_config.json"))
        files = os.listdir(subdir)
        metrics_files = sort_metrics_files([f for f in files if f.startswith("metrics") and f.endswith(".json")])
        if not metrics_files:
            continue

        for mf in metrics_files:
            metrics_path = os.path.join(subdir, mf)
            try:
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    metrics_data = json.load(fh)
            except Exception:
                continue
            flat = flatten_dict(metrics_data)

            for k in flat.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

            row_topic = "" if topic_written.get(subdir) else topic
            if row_topic and not topic_written.get(subdir):
                topic_written[subdir] = True

            row_agent = "" if agent_written.get(subdir) else (agent_label or "")
            if row_agent and not agent_written.get(subdir):
                agent_written[subdir] = True

            row = {
                "topic": row_topic,
                "agent_name": row_agent,
                "metrics_file": mf
            }
            row.update(flat)
            rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Metrics exported to {output_csv}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    collect_metrics("./results", "./results/all_metrics.csv")
