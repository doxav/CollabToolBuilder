import json
import numpy as np
from evaluate import load
from time import time
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def calculate_bibliography_coverage(target_bib: List[Dict], current_bib: List[Dict], embedding_name: str) -> float:
    """Calculate bibliography coverage using Hungarian algorithm."""
    if not target_bib or not current_bib:
        return 0.0
        
    # Get embeddings
    target_embeddings = [r[embedding_name] for r in target_bib]
    current_embeddings = [r[embedding_name] for r in current_bib]
    
    # Calculate distance matrix
    distance_matrix = 1 - cosine_similarity(target_embeddings, current_embeddings)
    
    # Use Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # Calculate coverage score (1 - normalized distance)
    total_distance = sum(distance_matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind)))
    coverage = 1 - (total_distance / len(target_bib))
    
    return coverage


def calculate_section_bibliography_coverage(
    target_section_refs: List[int],
    pred_section_refs: List[int],
    target_bib: List[Dict],
    pred_bib: List[Dict],
    embedding_name: str="resource_embedding_1"
) -> Dict:
    target_refs = [r for r in target_bib if r["resource_id"] in target_section_refs]
    pred_refs = [r for r in pred_bib if r["resource_id"] in pred_section_refs]
    score = calculate_bibliography_coverage(target_refs, pred_refs, embedding_name)
    return score

def _compare_documents(
    document: str | Path | Dict[str, Any],
    prediction: str | Path | Dict[str, Any],
    compare_on: str = "section",
) -> Dict[str, Any]:
    """Compare the 'compare_on' sections of document and prediction. Calculate MAUVE,
    and ROUGE-L scores on the actual text, and cosine similarity on the embeddings.

    Parameters
    ----------
    document : Dict[str, Any]
        Dictionary containing the grouth truth document. Must contain the keys
        'plan' and 'id'.
    prediction : Dict[str, Any]
        Dictionary containing the prediction to compare against. Must contain the keys
        'plan' and 'id'.
    compare_on : str, optional ['section', 'content']
        Whether to compare on the 'section' level i.e. the plan of the document, or
        the 'content' level.
    """
    if compare_on not in ["section", "content"]:
        raise ValueError(
            f"`compare_on` must be 'section' or 'content'. Received {compare_on}"
        )

    if isinstance(document, str) or isinstance(document, Path):
        with open(document, "r") as f:
            document = json.load(f)

    if isinstance(prediction, str) or isinstance(prediction, Path):
        with open(prediction, "r") as f:
            prediction = json.load(f)

    if not isinstance(document, dict) or not isinstance(prediction, dict):
        raise TypeError(
            "Both `document` and `prediction` must be dictionaries. Received "
            f"{type(document)} and {type(prediction)}"
        )

    if "plan" not in document or "plan" not in prediction:
        raise ValueError(
            f'Both `document` and `prediction` must contain the key "plan". At least '
            f"one of them does not."
        )

    start = time()
    doc1_name = f"ID: {document['id']} Title: {document['title']}"
    doc2_name = f"ID: {prediction['id']} Title: {prediction['title']}"
    logger.info(
        f"\n\tStarting to compare two documents on {compare_on}:"
        f"\n\t\t{doc1_name}"
        f"\n\t\t{doc2_name}"
    )

    mauve = load("mauve")
    rouge = load("rouge")

    section_results = []
    doc_plan: List[Dict[str, Any]] = document["plan"]
    predict_plan: List[Dict[str, Any]] = prediction["plan"]

    logger.info(
        f"\n\t{doc1_name} has {len(doc_plan)} sections."
        f"\n\t{doc2_name} has {len(predict_plan)} sections."
    )
    total_comparisons = min(len(doc_plan), len(predict_plan))
    # If plans have different lengths, just goes up to shortest
    for idx, (p_dict, d_dict) in enumerate(zip(predict_plan, doc_plan), start=1):
        # Compute MAUVE
        mauve_results = mauve.compute(
            predictions=[p_dict[compare_on]], references=[d_dict[compare_on]],verbose=True
        )
        mauve_score = mauve_results.mauve
        # Compute ROUGE-L
        results = rouge.compute(
            predictions=[p_dict[compare_on]],
            references=[d_dict[compare_on]],
            rouge_types=["rougeL"],
        )
        rouge_score = results["rougeL"]
        # Compute cosine distance between both section embeddings
        cosine_1 = cosine_similarity(
            [p_dict[f"{compare_on}_embedding_1"]], [d_dict[f"{compare_on}_embedding_1"]]
        )[0][0]
        cosine_2 = cosine_similarity(
            [p_dict[f"{compare_on}_embedding_2"]], [d_dict[f"{compare_on}_embedding_2"]]
        )[0][0]

        # Calculate bibliography coverage for both sections
        ref_coverage_1 = calculate_section_bibliography_coverage(
            d_dict.get("resources_used", []),
            p_dict.get("resources_used", []),
            document.get("resources", []),
            prediction.get("resources", []),
            "resource_embedding_1"
        )
        ref_coverage_2 = calculate_section_bibliography_coverage(
            d_dict.get("resources_used", []),
            p_dict.get("resources_used", []),
            document.get("resources", []),
            prediction.get("resources", []),
            "resource_embedding_2"
        )
        # Combine results
        result = {
            "section_id": idx,
            "mauve_similarity": mauve_score,
            "rouge_L_similarity": rouge_score,
            "embedding1_cosine_similarity": cosine_1,
            "embedding2_cosine_similarity": cosine_2,
            "bibliography_coverage1": ref_coverage_1,
            "bibliography_coverage2": ref_coverage_2,
        }
        section_results.append(result)
        logger.info(f"{idx}/{total_comparisons} sections compared.")

    # Calcualte total scores
    mauve_total = np.mean([x["mauve_similarity"] for x in section_results])
    rouge_total = np.mean([x["rouge_L_similarity"] for x in section_results])
    cosine_1_total = np.mean(
        [x["embedding1_cosine_similarity"] for x in section_results]
    )
    cosine_2_total = np.mean(
        [x["embedding2_cosine_similarity"] for x in section_results]
    )

    total_results = {
        "mauve_similarity": mauve_total,
        "rouge_L_similarity": rouge_total,
        "embedding1_cosine_similarity": cosine_1_total,
        "embedding2_cosine_similarity": cosine_2_total,
    }

    # Calculate global bibliography coverage using Hungarian algorithm
    global_bib_coverage_1 = calculate_bibliography_coverage(
        document.get("resources", []),
        prediction.get("resources", []),
        "resource_embedding_1"
    )
    global_bib_coverage_2 = calculate_bibliography_coverage(
        document.get("resources", []),
        prediction.get("resources", []),
        "resource_embedding_2"
    )
    if compare_on == "section":
        compare_on = "plan"

    output = {
        "document_id": document["id"],
        "prediction_id": prediction["id"],
        f"{compare_on}_total_similarity": total_results,
        f"{compare_on}_bysection_similarity": section_results,
        "bibliography_coverage1": global_bib_coverage_1,
        "bibliography_coverage2": global_bib_coverage_2
    }

    end = time()
    seconds = end - start
    mins = seconds / 60
    logger.info(
        f"\n\tFinished comparing document {compare_on}s:"
        f"\n\t\tThat took: {mins:.2f} mins ({seconds:.0f} seconds)"
    )
    return output

def compare_documents_sections(
    document1: str | Path | Dict[str, Any],
    document2: str | Path | Dict[str, Any],
) -> Dict[str, Any]:
    """This function takes two documents, a comparison method, compares the section
    headings (also called plans) of the documents in order using the specified method,
    and returns a dictionary containing the similarity scores.

    Definition: a document's 'plan' is the headings and subheadings of the document in
                order.

    Example Usage:
    >>> url_1 = 'https://en.wikipedia.org/wiki/Simulated_annealing'
    >>> url_2 = 'https://en.wikipedia.org/wiki/Dual-phase_evolution'
    >>> doc_1 = await extract_plan_and_content_wikipedia(url_1)
    >>> doc_2 = await extract_plan_and_content_wikipedia(url_2)
    >>> compare_plan = compare_documents_sections(doc_1, doc_2, None)
    """
    return _compare_documents(document1, document2, compare_on="section")

def compare_documents_content(
    document1: str | Path | Dict[str, Any],
    document2: str | Path | Dict[str, Any],
) -> Dict[str, Any]:
    """This function takes two documents, a comparison method, compares the sections
    of the documents using the specified method, and returns a dictionary containing
    the section-wise similarity scores.

    Definition: a document's 'content' is the text under the headings and subheadings

    Example Usage:
    >>> url_1 = 'https://en.wikipedia.org/wiki/Simulated_annealing'
    >>> url_2 = 'https://en.wikipedia.org/wiki/Dual-phase_evolution'
    >>> doc_1 = await extract_plan_and_content_wikipedia(url_1)
    >>> doc_2 = await extract_plan_and_content_wikipedia(url_2)
    >>> compare_sections = compare_documents_content(doc_1, doc_2, None)
    """
    # TODO - do we really need method? Or can we just do every metric every time?
    return _compare_documents(document1, document2, compare_on="content")
