import json
def retrieve_research_papers(search_query):
    """
The function retrieves research papers using the scholarly library by performing a search query and returning the titles, abstracts, and URLs of the top 5 results.
:param search_query: The query string used to search for research papers
:return: A JSON string containing the search success status and a list of papers with their titles, abstracts, and URLs if successful, or an error message if unsuccessful
"""
    try:
        # Check if scholarly is installed, otherwise install it
        import importlib.util
        if importlib.util.find_spec('scholarly') is None:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", 'scholarly'])
        
        # Import scholarly now that we've ensured it's installed
        from scholarly import scholarly
        
        # Perform the search
        search_results = scholarly.search_pubs(search_query)
        
        # Fetch some publications
        papers = []
        for result in search_results:
            # Fetch the publication to get the full data
            pub = scholarly.fill(result)
            paper = {
                'title': pub['bib']['title'],
                'abstract': pub['bib'].get('abstract', ''),
                'url': pub['pub_url'] if 'pub_url' in pub else ''
            }
            papers.append(paper)
            if len(papers) >= 5:  # Limit the number of papers to retrieve
                break
        
        return json.dumps({'success': True, 'papers': papers})
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})