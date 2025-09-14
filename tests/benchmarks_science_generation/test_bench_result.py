import markdown
import os
import sys
import json
import argparse
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from urllib.parse import urlparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from tests.benchmarks_science_generation.benchmark.doc_eval import evaluate_document

from tests.benchmarks_science_generation.benchmark.benchmark import get_solution_from_wikipedia, load_dea

import re


def md2latex(input_file: str, output_file: str):
    md_content = ""
    with open(input_file, "r") as f:
        md_content = f.read()
    if md_content:
        md = markdown.Markdown(extensions=["latex"])
        latex_out = md.convert(md_content)
        with open(output_file, "w") as f:
            f.write(latex_out)





def create_target_json(target_path: str, content: str):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Create a basic JSON structure with all required fields
    target_data = {
        "content": content,
        "metadata": {"title": "Generated Document", "type": "test_document"},
        "plan": [
            {
                "section": "Introduction",
                "title": "Introduction",
                "content": (
                    content[:500] if len(content) > 500 else content
                ),  # Use first 500 chars as sample
                "sections": [],
                "section_embedding_2": [0.0]
                * 768,  # Create a dummy embedding vector with correct dimension
                "content_embedding_2": [0.0]
                * 768,  # Create a dummy embedding vector with correct dimension
                "section_title_embedding_2": [0.0]
                * 768,  # Create a dummy embedding vector with correct dimension
                "resources_used": [],  # Add empty list for resources used
            }
        ],
        "resources": [],  # Empty list of resources
        "plan_embedding_2": [0.0]
        * 768,  # Create a dummy embedding vector with correct dimension
    }

    # Write the JSON file
    with open(target_path, "w") as f:
        json.dump(target_data, f, indent=2)

    return target_path





def display_metrics(methods_metrics):
    all_rows = []

    def flatten_metrics(method_name, metrics):
        rows = []

        # === FORMAT 1: Prometheus/WriteHere style ===
        if "prometheus_scores" in metrics:
            for k, v in metrics["prometheus_scores"].items():
                rows.append(
                    {
                        "Metric Group": "Prometheus Scores",
                        "Metric Name": k,
                        "p": None,
                        "r": None,
                        "f": v,
                        "Method": method_name,
                        "is_rouge": False,
                    }
                )

        if "writehere_scores" in metrics:
            for k, v in metrics["writehere_scores"].items():
                rows.append(
                    {
                        "Metric Group": "WriteHere Scores",
                        "Metric Name": k,
                        "p": None,
                        "r": None,
                        "f": v,
                        "Method": method_name,
                        "is_rouge": False,
                    }
                )

        if "article_metrics" in metrics:
            article = metrics["article_metrics"]
            if "entity_recall" in article:
                rows.append(
                    {
                        "Metric Group": "Article Metrics",
                        "Metric Name": "entity_recall",
                        "p": None,
                        "r": None,
                        "f": article["entity_recall"],
                        "Method": method_name,
                        "is_rouge": False,
                    }
                )
            if "rouge_scores" in article:
                for rouge_name, scores in article["rouge_scores"].items():
                    rows.append(
                        {
                            "Metric Group": "ROUGE Scores",
                            "Metric Name": rouge_name,
                            "p": scores.get("p"),
                            "r": scores.get("r"),
                            "f": scores.get("f"),
                            "Method": method_name,
                            "is_rouge": True,
                        }
                    )
            if "llm_scores" in article and article["llm_scores"]:
                for k, v in article["llm_scores"].items():
                    rows.append(
                        {
                            "Metric Group": "LLM Scores",
                            "Metric Name": k,
                            "p": None,
                            "r": None,
                            "f": v,
                            "Method": method_name,
                            "is_rouge": False,
                        }
                    )

        # === FORMAT 2: Report-style ===
        if "report" in metrics:
            report = metrics["report"]
            if "scores" in report:
                for k, v in report["scores"].items():
                    rows.append(
                        {
                            "Metric Group": "LLM Scores",
                            "Metric Name": k,
                            "p": None,
                            "r": None,
                            "f": v,
                            "Method": method_name,
                            "is_rouge": False,
                        }
                    )
            if "entity_recall" in report:
                rows.append(
                    {
                        "Metric Group": "Article Metrics",
                        "Metric Name": "entity_recall",
                        "p": None,
                        "r": None,
                        "f": report["entity_recall"],
                        "Method": method_name,
                        "is_rouge": False,
                    }
                )
            if "rouge_scores" in report:
                for rouge_name, scores in report["rouge_scores"].items():
                    rows.append(
                        {
                            "Metric Group": "ROUGE Scores",
                            "Metric Name": rouge_name,
                            "p": scores.get("p"),
                            "r": scores.get("r"),
                            "f": scores.get("f"),
                            "Method": method_name,
                            "is_rouge": True,
                        }
                    )
            if "coverage_speed" in report:
                for k, v in report["coverage_speed"].items():
                    rows.append(
                        {
                            "Metric Group": "Coverage Speed",
                            "Metric Name": k,
                            "p": None,
                            "r": None,
                            "f": v,
                            "Method": method_name,
                            "is_rouge": False,
                        }
                    )

        # ===  DEA Evaluation Metrics ===
        if "dea_evaluation_scores" in metrics:
            for k, v in metrics["dea_evaluation_scores"].items():
                rows.append(
                    {
                        "Metric Group": "DEA Evaluation Scores",
                        "Metric Name": k.split("(")[0],
                        "p": None,
                        "r": None,
                        "f": v,
                        "Method": method_name,
                        "is_rouge": False,
                    }
                )

        if "citation_quality" in metrics:
            rows.append(
                {
                    "Metric Group": "Citation",
                    "Metric Name": "citation_quality",
                    "p": None,
                    "r": None,
                    "f": metrics["citation_quality"],
                    "Method": method_name,
                    "is_rouge": False,
                }
            )

        return rows

    # Support nested method structure
    for method_name, metrics in methods_metrics.items():
        # If metrics is already a metric dict (has 'prometheus_scores', etc.), treat normally
        if any(
            k in metrics
            for k in [
                "prometheus_scores",
                "writehere_scores",
                "article_metrics",
                "dea_evaluation_scores",
            ]
        ):
            all_rows.extend(flatten_metrics(method_name, metrics))
        else:
            # Else, it's a nested dict like {'storm': {...}, 'costorm': {...}}
            for submethod, submetrics in metrics.items():
                full_method_name = f"{method_name}:{submethod}"
                all_rows.extend(flatten_metrics(full_method_name, submetrics))

    df = pd.DataFrame(all_rows)
    methods = df["Method"].unique().tolist()
    table_rows = []

    metric_index = df[["Metric Group", "Metric Name"]].drop_duplicates()

    for _, metric in metric_index.iterrows():
        mg = metric["Metric Group"]
        mn = metric["Metric Name"]
        row = {"Metric Group": mg, "Metric Name": mn}

        metric_rows = df[(df["Metric Group"] == mg) & (df["Metric Name"] == mn)]

        for method in methods:
            method_row = metric_rows[metric_rows["Method"] == method]
            if method_row.empty:
                val = ""
            else:
                is_rouge = method_row.iloc[0]["is_rouge"]
                if is_rouge:
                    p = method_row.iloc[0]["p"]
                    r = method_row.iloc[0]["r"]
                    f = method_row.iloc[0]["f"]

                    def fmt(x):
                        return f"{x:.3f}" if isinstance(x, float) else ""

                    val = f"p={fmt(p)} | r={fmt(r)} | f={fmt(f)}"
                else:
                    f = method_row.iloc[0]["f"]
                    val = f"{f:.3f}" if isinstance(f, float) else (f if f else "")

            row[method] = val

        table_rows.append(row)

    combined = pd.DataFrame(table_rows)
    combined["Metric Group"] = combined["Metric Group"].mask(
        combined["Metric Group"].duplicated(), ""
    )

    print(tabulate(combined, headers="keys", tablefmt="fancy_grid", showindex=False))
    print(f"{'='*20} Legend {'='*20}")
    print("p = precision, r = recall, f = F1 score")





def get_absolute_path(relative_path):
    """
    Converts a relative path to an absolute path based on the current script's directory.

    Parameters
    ----------
    relative_path : str
        The relative path to convert.

    Returns
    -------
    str
        The absolute path.
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(SCRIPT_DIR, relative_path))


def extract_id_and_date(folder_name):
    """
    Extracts the ID and date from a folder name in the format:
    'ID_YYYYMMDD-HHMMSS_optionalText'.

    Parameters
    ----------
    folder_name : str
        The folder name to extract from.

    Returns
    -------
    tuple or None
        A tuple (id_str, date_str) if matched, else None.
    """
    pattern = re.compile(r"^(\d{7})_(\d{8}-\d{6})")
    match = pattern.match(folder_name)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def get_folder_by_id(path, id_str):
    """
    Searches for a folder in the given directory that has the specified ID in the
    format 'ID_YYYYMMDD-HHMMSS_optionalText'.

    Parameters
    ----------
    path : str
        The path to the parent directory containing dated folders.
    id_str : str
        The ID to search for at the beginning of the folder name.

    Returns
    -------
    str
        The name of the matching folder.

    Raises
    ------
    FileNotFoundError
        If the path is not a valid directory.
    ValueError
        If no matching folder is found.
    """
    folder = path.split("/")[-1]
    path = get_absolute_path(path)

    if not os.path.isdir(path):
        raise FileNotFoundError(f"The path '{path}' is not a valid directory.")
    id_num = int(id_str)
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            result = extract_id_and_date(entry)
            if result:
                extracted_id, _ = result
                if extracted_id == id_num:
                    return os.path.join(path, entry)

    raise ValueError(f"No folder found with ID '{id_str}' in {folder}")


def get_file_path_in_folder(folder_path, filename):
    """
    Recursively searches for a file inside the given folder and its subdirectories.
    Returns the full path to the file if found, otherwise raises an exception.

    Parameters
    ----------
    folder_path : str
        Full path to the root folder where the search should begin.
    filename : str
        Name of the file to search for (exact match).

    Returns
    -------
    str
        Full path to the found file.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or the file is not found.

    Examples
    --------
    >>> get_file_path_in_folder('/backups/20250627-100317_backup', 'data.json')
    '/backups/20250627-100317_backup/some/subdir/data.json'
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    for root, dirs, files in os.walk(folder_path):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(
        f"The file '{filename}' was not found in '{folder_path}' or its subdirectories."
    )


def folder_contains_file(folder_path: str, filename: str) -> bool:
    """
    Recursively checks if `filename` exists anywhere under `folder_path`.
    """
    for root, _, files in os.walk(folder_path):
        if filename in files:
            return True
    return False


def get_last_id(base_path: str, filename: str) -> int:
    """
    Scans subdirectories in `base_path` with names starting with a numeric prefix (e.g., '0000001_'),
    checks if they or any of their subdirectories contain `filename`, and returns the highest numeric ID.

    Args:
        base_path (str): The base directory to search within.
        filename (str): The file that must exist somewhere under the directory.

    Returns:
        int: The highest numeric ID among directories containing the specified file.
    """
    base_path = get_absolute_path(base_path)
    id_pattern = re.compile(r"^(\d+)_")

    max_id = 0

    for entry in os.scandir(base_path):
        if entry.is_dir():
            match = id_pattern.match(entry.name)
            if match:
                full_dir_path = os.path.join(base_path, entry.name)
                if folder_contains_file(full_dir_path, filename):
                    num_id = int(match.group(1))
                    max_id = max(max_id, num_id)

    return max_id


def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the arguments parsed.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Evaluate benchmark results.")
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic to generate the article.",
        default="Taylor Hawkins",
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Unique identifier common to all files in the benchmark folder.",
        default=None,
    )
    parser.add_argument(
        "--id-storm",
        type=str,
        help="Unique identifier for the Storm article.",
        default=None,
    )
    parser.add_argument(
        "--id-costorm",
        type=str,
        help="Unique identifier for the CoStorm report.",
        default=None,
    )
    parser.add_argument(
        "--id-writehere",
        type=str,
        help="Unique identifier for the WriteHere report.",
        default=None,
    )
    parser.add_argument(
        "--results-path",
        type=str,
        help="Path to the results folder.",
        default="../benchmark/results",
    )
    parser.add_argument(
        "--storm-path",
        type=str,
        help="Path to the Storm article markdown file.",
        default="tmp_storm",
    )
    parser.add_argument(
        "--storm-file",
        type=str,
        help="Name of the Storm article markdown file.",
        default="storm_gen_article.txt",
    )
    parser.add_argument(
        "--storm-turns-file",
        type=str,
        help="Name of the Storm article turns file.",
        default="conversation_log.json",
    )
    parser.add_argument(
        "--costorm-path",
        type=str,
        help="Path to the CoStorm report markdown file.",
        default="tmp_costorm",
    )
    parser.add_argument(
        "--costorm-file",
        type=str,
        help="Name of the CoStorm report markdown file.",
        default="report.md",
    )
    parser.add_argument(
        "--costorm-turns-file",
        type=str,
        help="Name of the CoStorm report turns file.",
        default="instance_dump.json",
    )
    parser.add_argument(
        "--writehere-path",
        type=str,
        help="Path to the WriteHere report markdown file.",
        default="tmp_writehere",
    )
    parser.add_argument(
        "--writehere-file",
        type=str,
        help="Name of the WriteHere report markdown file.",
        default="output.jsonl",
    )
    parser.add_argument(
        "--writehere-turns-file",
        type=str,
        help="Name of the WriteHere report turns file.",
        default="turns.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use for evaluation (Wikipedia,DEA).",
        choices=["Wikipedia", "DEA"],
        default="Wikipedia",
    )

    return parser.parse_args()

def json_to_references_strom(data):
    """Convert JSON dump from storm to Wikipedia-style markdown references"""
    if isinstance(data, str):
        data = json.loads(data) if data.startswith('{') else json.load(open(data))
    
    refs = ["## References\n"]
    for i, (url, info) in enumerate(sorted(data.get('url_to_info', {}).items()), 1):
        title = info.get('title', 'Unknown')
        desc = info.get('description', '').split(' — ', 1)[-1] if ' — ' in info.get('description', '') else info.get('description', '')
        domain = urlparse(url).netloc.replace('www.', '').split('.')[0].title()
        
        ref = f"{i}. [{title}]({url})"
        if domain not in title: ref += f" - {domain}"
        if desc and len(desc) < 150: ref += f". {desc}"
        refs.append(ref)
    
    return '\n'.join(refs)

def storm_prossessing():
    """
    Process the Storm files and prepare the paths for evaluation.
    """
    RESULT_FOLDER_STORM = os.path.join(RESULT_FOLDER, args.storm_path)
    WANTED_FILE_STORM = args.storm_file
    TURN_FILE_STORM = args.storm_turns_file
    storm_id = args.id_storm or SHARED_ID or get_last_id(RESULT_FOLDER_STORM, WANTED_FILE_STORM)
    
    result_storm = get_folder_by_id(RESULT_FOLDER_STORM, storm_id)
    storm_md = get_file_path_in_folder(result_storm, WANTED_FILE_STORM)
    storm_turns = get_file_path_in_folder(result_storm, TURN_FILE_STORM)

    article = Path(storm_md).read_text(encoding="utf-8")
    # add json_to_references from url_to_info.json
    url_to_info_path = get_file_path_in_folder(result_storm, "url_to_info.json")
    if os.path.exists(url_to_info_path):
        article += "\n\n"+json_to_references_strom(url_to_info_path)

    turns = json.loads(Path(storm_turns).read_text(encoding="utf-8"))
    return article, turns


def json_to_references_costorm(data,article):
    """Convert JSON dump from CoStorm to Wikipedia-style markdown references"""
    if isinstance(data, str):
        data = json.loads(data) if data.startswith('{') else json.load(open(data))
    ref_dict = data["knowledge_base"]["info_uuid_to_info_dict"]
    
    refs = ["## References\n"]
    for data in ref_dict.values():
        i = data.get('citation_uuid', 'Unknown')
        if not (f"[{i}]." in article or f"[{i}][" in article):
            continue
        title = data.get('title', 'Unknown')
        desc = data.get('description', '').split(' — ', 1)[-1] if ' — ' in data.get('description', '') else data.get('description', '')
        url = data.get('url', '')
        domain = urlparse(url).netloc.replace('www.', '').split('.')[0].title()
        
        ref = f"{i}. [{title}]({url})"
        if domain not in title: ref += f" - {domain}"
        if desc and len(desc) < 150: ref += f". {desc}"
        refs.append(ref)
    
    return '\n'.join(refs)

def costorm_prossessing():
    """
    Process the CoStorm files and prepare the paths for evaluation.
    """
    RESULT_FOLDER_COSTORM = os.path.join(RESULT_FOLDER, args.costorm_path)
    WANTED_FILE_COSTORM = args.costorm_file
    TURN_FILE_COSTORM = args.costorm_turns_file
    URL_FILE_COSTORM = "instance_dump.json"
    
    costorm_id = args.id_costorm or SHARED_ID or get_last_id(RESULT_FOLDER_COSTORM, WANTED_FILE_COSTORM)
    
    result_costorm = get_folder_by_id(RESULT_FOLDER_COSTORM, costorm_id)
    costorm_md = get_file_path_in_folder(result_costorm, WANTED_FILE_COSTORM)
    costorm_turns = get_file_path_in_folder(result_costorm, TURN_FILE_COSTORM)
    url_file = get_file_path_in_folder(result_costorm, URL_FILE_COSTORM)
    article = Path(costorm_md).read_text(encoding="utf-8")
    if os.path.exists(url_file):
        article += "\n\n" + json_to_references_costorm(url_file,article)
    turns = json.loads((Path(costorm_turns)).read_text(encoding="utf-8"))
    return article, turns


def writehere_prossessing():
    """
    Process the WriteHere files and prepare the paths for evaluation.
    """
    RESULT_FOLDER_WRITEHERE = os.path.join(RESULT_FOLDER, args.writehere_path)
    WANTED_FILE_WRITEHERE = args.writehere_file
    TURN_FILE_WRITEHERE = args.writehere_turns_file
    writehere_id = (
        args.id_writehere
        or SHARED_ID
        or get_last_id(RESULT_FOLDER_WRITEHERE, WANTED_FILE_WRITEHERE)
    )
    result_writehere = get_folder_by_id(RESULT_FOLDER_WRITEHERE, writehere_id)
    writehere_json = get_file_path_in_folder(result_writehere, WANTED_FILE_WRITEHERE)
    writehere_md = writehere_json.replace(".jsonl", ".md")
    writehere_turns = get_file_path_in_folder(result_writehere, TURN_FILE_WRITEHERE)
    if not os.path.exists(writehere_md):
        # Convert JSONL to Markdown
        with open(writehere_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = data["result"]
        with open(writehere_md, "w", encoding="utf-8") as f:
            f.write(result)
    article = Path(writehere_md).read_text(encoding="utf-8")
    turns = json.loads((Path(writehere_turns)).read_text(encoding="utf-8"))
    return article, turns




def get_solution_dea():
    path = get_absolute_path("../benchmark/dea")
    iterator_dea = load_dea(path)
    topic, itent, solution = next(iterator_dea)
    return topic, itent, solution



if __name__ == "__main__":

    args = _build_parser()
    EVALUATE_METHOD = evaluate_document
    # First convert markdown to LaTeX
    RESULT_FOLDER = args.results_path
    if args.dataset == "FreshWiki":
        intent, solution = get_solution_from_wikipedia(args.topic)
    elif args.dataset == "DEA":
        topic, intent, solution = get_solution_dea()

    # TODO: CoStormGraph paths LATER
    # TODO: Solution paths (FeshWiki => Subject + Wikipedia) / after DEA: https://github.com/doxav/document_embedding_analysis/tree/main/output/latex title/topic:abstract output:https://github.com/doxav/document_embedding_analysis/tree/main/data/latex)

    SHARED_ID = args.id

    storm_article, storm_turns = storm_prossessing()

    costorm_article, costorm_turns = costorm_prossessing()

    writehere_article, writehere_turns = writehere_prossessing()
    # TODO: CoStormGraph costorm_graph_md

    print("====Doc eval====")
    metrics_storm = EVALUATE_METHOD(storm_article, storm_turns, solution)

    metrics_costorm = EVALUATE_METHOD(costorm_article, costorm_turns, solution)
    metrics_writehere = EVALUATE_METHOD(writehere_article, writehere_turns, solution)

    print("====Doc eval done====")
    result = {
        "storm": metrics_storm,
        "costorm": metrics_costorm,
        "writehere": metrics_writehere,
    }

    display_metrics(result)

    # TODO: Synthetize results in a table
