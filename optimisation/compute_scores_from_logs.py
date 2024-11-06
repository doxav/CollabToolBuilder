import glob
import re
import ast
import pandas as pd
from datetime import datetime
from itertools import islice
import argparse
import os

def process_in_chunks_with_generator(file_match='websocketdata_*', chunk_size=1000, prefix='scores/final_scores_'):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    pattern = re.compile(r'Performance scores: (\[.*?\])', re.DOTALL)

    # Get all files matching the pattern
    matching_files = glob.glob(file_match)
    if not matching_files:
        print("No files found matching the pattern.")
        return

    # Première passe pour compter le nombre total de correspondances
    total_matches = 0
    for filename in matching_files:
        with open(filename, 'r') as file:
            total_matches += len(pattern.findall(file.read()))
    print(f"Total matches found across all files: {total_matches}")

    def score_generator(filename):
        with open(filename, 'r') as file:
            for match in pattern.finditer(file.read()):
                try:
                    performance_scores = ast.literal_eval(match.group(1))
                    for perf_score_dict in performance_scores:
                        values = [v for v in perf_score_dict.values() if isinstance(v, (int, float))]
                        if values:
                            #print(values)
                            avg_score = sum(values) / len(values)
                            final_score = 0.5 + 10 * avg_score + 20 * (1 + avg_score)
                            yield {
                                'file': filename,
                                'score': final_score,
                                'plan similarity': perf_score_dict.get('plan/titles similarity (top:1, worst:0)', 0),
                                'section similarity': perf_score_dict.get('sections contents similarity (top:1, worst:0)', 0),
                                'sections count length': perf_score_dict.get('sections count (top:1, <1:too short, >1:too long)', 0)
                            }
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing performance scores in file {filename}: {e}")

    # Liste pour accumuler tous les résultats finaux
    all_scores = []

    # Traitement par fichier
    for filename in matching_files:
        print(f"Processing file: {filename}")
        chunks = []
        processed_matches = 0
        generator = score_generator(filename)
        
        # Traitement par blocs
        while True:
            chunk = list(islice(generator, chunk_size))
            if not chunk:
                break
            df_chunk = pd.DataFrame(chunk)
            chunks.append(df_chunk)
            
            # Mise à jour de la progression
            processed_matches += len(chunk)
            print(f"Processed {processed_matches} matches in {filename}")

        # Concaténer les chunks pour le fichier actuel et enregistrer
        if chunks:
            file_scores = pd.concat(chunks, ignore_index=True)
            file_scores.sort_values(by='score', ascending=False, inplace=True)
            all_scores.append(file_scores)  # Ajouter aux scores globaux
            # get max from each column
            max_values = file_scores.max()
            print(f"Max values for {filename}:")
            print(max_values)
            
            # Sauvegarder les résultats du fichier actuel
            output_file = f'{prefix}{os.path.basename(filename).split(".")[0]}_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
            file_scores.to_csv(output_file, index=False)
            print(f"Scores for {filename} saved to {output_file}")

    # Création du fichier agrégé final uniquement s'il y a plusieurs fichiers
    if len(matching_files) > 1 and all_scores:
        base_name = os.path.basename(file_match).split('.')[0]
        aggregated_file = f'{prefix}{base_name}_aggregated_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
        final_scores = pd.concat(all_scores, ignore_index=True)
        final_scores.sort_values(by='score', ascending=False, inplace=True)
        max_values = final_scores.max()
        print(f"Max values for {aggregated_file}:")
        print(max_values)
        
        # Generate an aggregated file name with the base of file_match
        final_scores.to_csv(aggregated_file, index=False)
        print(f"Aggregated final scores saved to file: {aggregated_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and summarize performance scores.")
    parser.add_argument("--file_pattern", type=str, default="websocketdata_*", help="File pattern to match logs")
    args = parser.parse_args()
    
    process_in_chunks_with_generator(args.file_pattern)