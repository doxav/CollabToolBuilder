import json
import argparse
from pathlib import Path
from .benchmark import evaluate_document_and_save_metrics


FOLDER = Path("benchmark/results")


def process_storm_dir(folder: Path,topic:str, force_evaluation: bool = False):
    """
    Process the storm directory to evaluate documents and save metrics.
    """
    for f in folder.iterdir():
        if f.is_dir():
            sub_dir = [f for f in f.iterdir() if f.is_dir()][0].name
            print(f"Processing storm folder: {f}")
            metrics_file = f / f"{sub_dir}/metrics.json"
            print(f"Checking for metrics file: {metrics_file}")
            if metrics_file.exists():
                if not force_evaluation:
                    print(f"Skipping {f} as metrics.json already exists.")
                    continue
                else:
                    print(f"Force evaluation for {f}.")
                    metrics_file.unlink()
            if not (f / f"{sub_dir}/storm_gen_article.txt").exists() or not (f / f"{sub_dir}/conversation_history.json").exists():
                print(f"Skipping {f} as required files are missing.")
                continue      
            report = (f / f"{sub_dir}/storm_gen_article.txt").read_text()
            turns = (f / f"{sub_dir}/conversation_history.json").read_text()
            evaluate_document_and_save_metrics(
                topic=topic,
                report=report,
                turns=turns,
                folder=f / f"{sub_dir}",
            )
def process_costorm_dir(folder: Path,topic:str, force_evaluation: bool = False):
    """
    Process the costorm directory to evaluate documents and save metrics.
    """
    for f in folder.iterdir():
        if f.is_dir():
            print(f"Processing costorm folder: {f}")
            metrics_file = f / "/metrics.json"
            if metrics_file.exists():
                if not force_evaluation:
                    print(f"Skipping {f} as metrics.json already exists.")
                    continue
                else:
                    print(f"Force evaluation for {f}.")
                    metrics_file.unlink()
            if not (f / "report.md").exists() or not (f / "instance_dump.json").exists():
                print(f"Skipping {f} as required files are missing.")
                continue
            report = (f / "report.md").read_text()
            turns = (f / "instance_dump.json").read_text()
            evaluate_document_and_save_metrics(
                topic=topic,
                report=report,
                turns=turns,
                folder=f,
            )
def process_writehere_dir(folder: Path,topic:str, force_evaluation: bool = False):
    """
    Process the writehere directory to evaluate documents and save metrics.
    """
    for f in folder.iterdir():
        if f.is_dir():
            print(f"Processing writehere folder: {f}")
            metrics_file = f / "/metrics.json"
            if metrics_file.exists():
                if not force_evaluation:
                    print(f"Skipping {f} as metrics.json already exists.")
                    continue
                else:
                    print(f"Force evaluation for {f}.")
                    metrics_file.unlink()
            if not (f / "output.jsonl").exists() or not (f / "turns.json").exists():
                print(f"Skipping {f} as required files are missing.")
                continue
            with (f / "output.jsonl").open("r", encoding="utf-8") as file_report:
                report = json.load(file_report)
            report = report["result"]
            with (f / "turns.json").open("r", encoding="utf-8") as file_turns:
                turns = json.load(file_turns)
            evaluate_document_and_save_metrics(
                topic=topic,
                report=report,
                turns=turns,
                folder=f,
            )
            
            
def build_args():
    """
    Build command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Process storm, costorm, and writehere directories.")
    parser.add_argument("--force-evaluation", action="store_true", help="Force evaluation even if metrics.json exists.")
    parser.add_argument("--force-evaluation-storm", action="store_true", help="Force evaluation for storm even if metrics.json exists.")
    parser.add_argument("--force-evaluation-costorm", action="store_true", help="Force evaluation for costorm even if metrics.json exists.")
    parser.add_argument("--force-evaluation-writehere", action="store_true", help="Force evaluation for writehere even if metrics.json exists.")
    parser.add_argument("--topic", type=str, default="Taylor Hawkins", help="Topic to use for evaluation.")
    parser.add_argument("--storm-folder", type=str, default="tmp_storm", help="Path to the storm directory.")
    parser.add_argument("--costorm-folder", type=str, default="tmp_costorm", help="Path to the costorm directory.")
    parser.add_argument("--writehere-folder", type=str, default="tmp_writehere", help="Path to the writehere directory.")
    return parser.parse_args()

def main():
    """
    Main function to process directories based on command line arguments.
    """
    args = build_args()
    
    FORCE_EVALUATION = args.force_evaluation
    FORCE_EVALUATION_STORM = args.force_evaluation_storm or FORCE_EVALUATION
    FORCE_EVALUATION_COSTORM = args.force_evaluation_costorm or FORCE_EVALUATION
    FORCE_EVALUATION_WRITEHERE = args.force_evaluation_writehere or FORCE_EVALUATION
    
    TOPIC = args.topic
    STORM_FOLDER = FOLDER / args.storm_folder
    COTORM_FOLDER = FOLDER / args.costorm_folder
    WRITEHERE_FOLDER = FOLDER / args.writehere_folder

    print("Processing storm directory...")
    process_storm_dir(STORM_FOLDER, TOPIC, force_evaluation=FORCE_EVALUATION_STORM)
    print("Processing costorm directory...")
    process_costorm_dir(COTORM_FOLDER, TOPIC, force_evaluation=FORCE_EVALUATION_COSTORM)
    print("Processing writehere directory...")
    process_writehere_dir(WRITEHERE_FOLDER, TOPIC, force_evaluation=FORCE_EVALUATION_WRITEHERE)
    print("All directories processed.")
    
    
if __name__ == "__main__":
    main()