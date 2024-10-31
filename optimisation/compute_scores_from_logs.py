import glob
import re
import ast

"""
def compute_scores():
    import glob
    import re
    import ast

    # Initialize a list to store the final scores
    final_scores = []

    # Use glob to get all files starting with 'websocketdata_'
    for filename in glob.glob('websocketdata_*'):
        with open(filename, 'r') as file:
            content = file.read()

            # Find all occurrences of 'Performance scores: [...]' in the content
            matches = re.findall(r'Performance scores: (\[.*?\])', content, re.DOTALL)

            # Process each matched performance score
            for match in matches:
                try:
                    # Safely evaluate the list of dictionaries
                    performance_scores = ast.literal_eval(match)

                    # Process each performance score dictionary
                    for perf_score_dict in performance_scores:
                        # Extract all numerical values from the dictionary
                        values = list(perf_score_dict.values())

                        # Compute the average of the values
                        if values:
                            best_score_without_validation = sum(values) / len(values)
                            validated_score_avg = best_score_without_validation
                            percentage_no_runtime_error = 0.5

                            # Compute the final score using the formula
                            final_score = (
                                percentage_no_runtime_error +
                                10 * best_score_without_validation +
                                20 * (1 + validated_score_avg)
                            )

                            # Store the result with file information
                            final_scores.append({
                                'filename': filename,
                                'performance_score': perf_score_dict,
                                'average': best_score_without_validation,
                                'final_score': final_score
                            })
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing performance scores in file {filename}: {e}")

    # Output the final scores
    for result in final_scores:
        print(f"File: {result['filename']}")
        print(f"Performance Score Dict: {result['performance_score']}")
        print(f"Average Score: {result['average']:.6f}")
        print(f"Final Score: {result['final_score']:.6f}")
        print('-' * 50)

# Call the function
compute_scores()
"""

import glob
import re
import ast
import pandas as pd

def compute_and_summarize_scores():
    import glob
    import re
    import ast
    import pandas as pd

    # Initialize a list to store the final scores
    final_scores = []

    # Use glob to get all files starting with 'websocketdata_'
    for filename in glob.glob('websocketdata_*'):
        with open(filename, 'r') as file:
            lines = file.readlines()
            content = ''.join(lines)

            # Find all occurrences of 'Performance scores: [...]' in the content along with their line numbers
            pattern = re.compile(r'Performance scores: (\[.*?\])', re.DOTALL)
            matches = pattern.finditer(content)

            for match in matches:
                try:
                    # Extract the matched text
                    perf_scores_text = match.group(1)

                    # Get the position to calculate line number
                    start_pos = match.start()

                    # Calculate the line number where the performance score was found
                    line_in_log = content.count('\n', 0, start_pos) + 1

                    # Safely evaluate the list of dictionaries
                    performance_scores = ast.literal_eval(perf_scores_text)

                    # Process each performance score dictionary
                    for perf_score_dict in performance_scores:
                        # Extract required metrics
                        plan_similarity = perf_score_dict.get('plan/titles similarity (top:1, worst:0)', 0)
                        section_similarity = perf_score_dict.get('sections contents similarity (top:1, worst:0)', 0)
                        sections_content_length = perf_score_dict.get('sections contents length (top:1, <1:too short, >1:too long)', 0)

                        # Extract all numerical values from the dictionary for average calculation
                        values = [v for v in perf_score_dict.values() if isinstance(v, (int, float))]

                        # Compute the average of the values
                        if values:
                            best_score_without_validation = sum(values) / len(values)
                            validated_score_avg = best_score_without_validation
                            percentage_no_runtime_error = 0.5

                            # Compute the final score using the formula
                            final_score = (
                                percentage_no_runtime_error +
                                10 * best_score_without_validation +
                                20 * (1 + validated_score_avg)
                            )

                            # Store the result with required information
                            final_scores.append({
                                'file': filename,
                                'score': final_score,
                                'plan similarity': plan_similarity,
                                'section similarity': section_similarity,
                                'sections content length': sections_content_length,
                                'line in log': line_in_log
                            })
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing performance scores in file {filename}: {e}")

    # Create a DataFrame from the results
    df = pd.DataFrame(final_scores)

    # Sort the DataFrame by 'score' in descending order
    df_sorted = df.sort_values(by='score', ascending=False)

    # Reset the index
    df_sorted.reset_index(drop=True, inplace=True)

    # Display the table
    print(df_sorted.to_string(index=False))

# Call the function
compute_and_summarize_scores()
