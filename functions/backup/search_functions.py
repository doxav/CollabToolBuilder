def search_wikipedia(term):
    """ Searches for a given term on Wikipedia and returns a summary of the page if it exists.
    :param term: The term to search for
    :return: A summary of the Wikipedia page
    """
    import wikipediaapi
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(term)
    return page.summary if page.exists() else "No page found for the term."

def search_arxiv(term):
    """ Searches the arXiv database for papers containing the given search term.
    :param term: The term to search for
    :return: A list of dictionaries containing the title and summary of each paper
    """
    import arxiv
    search = arxiv.Search(
      query = term,
      max_results = 10,
      sort_by = arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append({'title': result.title, 'summary': result.summary})
    return papers if papers else "No papers found for the term."

