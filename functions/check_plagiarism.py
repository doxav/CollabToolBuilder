import json
import subprocess
import sys
from difflib import SequenceMatcher
from env.IR_CPS_TechSynthesis.env import SynthesisManager

def install_and_import(package):
    """ Helper function to install and import packages dynamically """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = __import__(package)
    except Exception as e:
        print(f"Error installing {package}: {str(e)}")

def retrieve_research_papers(search_query):
    """
    Retrieves research papers related to the search query.

    Args:
    search_query (str): The query string for searching research papers.

    Returns:
    dict: JSON containing search results or error message.
    """
    try:
        scholarly = __import__('scholarly')
    except ImportError:
        install_and_import('scholarly')
        scholarly = __import__('scholarly')

    try:
        search_results = scholarly.search_pubs(search_query)
        papers = []
        for i, result in enumerate(search_results):
            if i >= 5:  # Limit the number of papers
                break
            pub = scholarly.fill(result)
            paper = {
                'title': pub['bib']['title'],
                'abstract': pub['bib'].get('abstract', ''),
                'url': pub.get('pub_url', '')
            }
            papers.append(paper)
        return {'success': True, 'papers': papers}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def check_plagiarism(bot: SynthesisManager, title: str, abstract: str, document_id: str):
    """
    Checks plagiarism by comparing the given abstract with research papers.

    Args:
    bot (SynthesisManager): Bot instance to interact with documents.
    title (str): Title of the document.
    abstract (str): Abstract of the document.
    document_id (str): ID of the document.

    Returns:
    dict: Plagiarism score and additional information.
    """
    result = retrieve_research_papers(title)
    if not result['success']:
        return {'success': False, 'error': result['error']}

    plag_scores = []
    for paper in result['papers']:
        sim_score = SequenceMatcher(None, abstract, paper['abstract']).ratio() * 100
        plag_scores.append(sim_score)

    avg_plag_score = sum(plag_scores) / len(plag_scores) if plag_scores else 0
    plagiarism_info = {
        'success': True,
        'plagiarism_score': avg_plag_score,
        'related_papers': result['papers']
    }

    # Log the plagiarism check event
    section_id = bot.create_and_add_section_then_return_id("Plagiarism Check Results", json.dumps(plagiarism_info, indent=4))
    bot.add_event("plagiarism_check", {"document_id": document_id, "section_id": section_id})

    return plagiarism_info