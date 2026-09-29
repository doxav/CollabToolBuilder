async def fetch_full_content(ar5iv_link):
    """
    Asynchronously fetch the full HTML content of the paper from ar5iv and clean it.
    Extract only the visible text and keep simple links in the format: "Text (URL)".
    Ensures that multiple newlines are reduced to a single newline.
    """
    import aiohttp
    from bs4 import BeautifulSoup
    import re

    async with aiohttp.ClientSession() as session:
        async with session.get(ar5iv_link) as response:
            if response.status == 200:
                full_content = await response.text()

                # Parse the HTML and clean it
                soup = BeautifulSoup(full_content, "html.parser")

                # Remove script and style tags
                for script in soup(["script", "style"]):
                    script.extract()

                # Keep only visible text and convert <a> tags to "text (URL)"
                def get_visible_text(element):
                    visible_texts = []
                    for tag in element.find_all(True):
                        if tag.name == "a" and tag.get("href"):
                            # Convert links to the format "text (URL)"
                            link_text = tag.get_text()
                            href = tag.get("href")
                            visible_texts.append(f"{link_text} ({href})")
                        else:
                            visible_texts.append(tag.get_text())
                    return ' '.join(visible_texts).strip()

                # Extract only the visible text
                clean_text = get_visible_text(soup)

                # Remove multiple newlines and ensure only one newline between sections
                clean_text = re.sub(r'\n+', '\n', clean_text)

                return clean_text
            else:
                return 'Unable to fetch full paper content in HTML from ar5iv'

async def search_arxiv(query, output_format='json', max_results=10, fetch_full_paper=False):
    """
    Search arXiv for articles relating to `query`.
    Returns a list of dictionaries containing article information, with ar5iv HTML links.
    Downloads the abstract and optionally the full paper content in HTML format from ar5iv.
    """
    import requests
    import feedparser
    import asyncio

    # URL for querying the arXiv API
    arxiv_url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending'

    # Make the request to the API for the abstract
    response = requests.get(arxiv_url)

    # Check if the request was successful
    if response.status_code != 200:
        return []

    # Parse the response text with feedparser
    feed = response.text
    feed = feedparser.parse(feed)

    # Parse and transform the results
    results = []
    fetch_tasks = []  # List to hold asyncio tasks for fetching full papers

    for entry in feed['entries']:
        # Extract the arXiv ID from the 'id' field in the entry (it's the identifier for the paper)
        arxiv_id = entry.get('id', '').split('/abs/')[-1]  # Extract the arXiv ID
        
        # Generate the ar5iv link for the full paper in HTML
        ar5iv_link = f'https://ar5iv.labs.arxiv.org/html/{arxiv_id}'

        # Store abstract data
        paper_data = {
            'title': entry.get('title', ''),
            'ar5iv_link': ar5iv_link,
            'abstract': entry.get('summary', ''),  # Explicitly using 'abstract'
            'full_content': 'Not fetched'  # Will update if full paper content is fetched
        }

        # Add task to fetch full content if fetch_full_paper is True
        if fetch_full_paper:
            fetch_tasks.append((paper_data, fetch_full_content(ar5iv_link)))

        results.append(paper_data)

    # If fetch_full_paper is True, gather all async tasks to download the full content
    if fetch_full_paper and fetch_tasks:
        # Execute all fetch tasks concurrently
        full_contents = await asyncio.gather(*[task[1] for task in fetch_tasks])

        # Update the results with full HTML content
        for idx, content in enumerate(full_contents):
            fetch_tasks[idx][0]['full_content'] = content

    return results

# Synchronous wrapper to make the function easier to call in a typical script
def search_arxiv_sync(query, output_format='json', max_results=10, fetch_full_paper=False):
    """
    Synchronous wrapper around the async search_arxiv function.
    """
    import asyncio
    return asyncio.run(search_arxiv(query, output_format=output_format, max_results=max_results, fetch_full_paper=fetch_full_paper))

def search_openalex(query: str, max_results: int = 10, year_from: int = None, year_to: int = None,
                    language: str = None, open_access_only: bool = False, mailto: str = None):
    """Retrieve academic papers from OpenAlex (250M+ works, incl. HAL, arXiv, publishers) for a query.

    Args:
        query: free-text query searched in titles, abstracts and full texts, e.g. "social media violence".
        max_results: maximum number of works to return (1 to 200).
        year_from: keep works published from this year (e.g. 2022), inclusive.
        year_to: keep works published up to this year, inclusive.
        language: ISO 639-1 code of the work language, e.g. "fr" or "en".
        open_access_only: if True, keep only open-access works.
        mailto: optional contact e-mail, gives access to OpenAlex's faster "polite pool".

    Returns:
        A list of dicts with keys: title, authors, year, abstract, doi, url, oa_url,
        source, type, language, cited_by_count, openalex_id.
    """
    import requests

    filters = ["is_paratext:false"]  # Exclude non-research content
    if year_from:
        filters.append(f"from_publication_date:{int(year_from)}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{int(year_to)}-12-31")
    if language:
        filters.append(f"language:{language}")
    if open_access_only:
        filters.append("is_oa:true")

    params = {
        "search": query,
        "filter": ",".join(filters),
        "sort": "relevance_score:desc",
        "per_page": max(1, min(int(max_results), 200)),
    }
    if mailto:
        params["mailto"] = mailto

    def rebuild_abstract(inverted_index):
        # OpenAlex ships abstracts as {word: [positions]}; rebuild the plain text.
        if not inverted_index:
            return "No abstract available"
        positions = [(pos, word) for word, pos_list in inverted_index.items() for pos in pos_list]
        return " ".join(word for _, word in sorted(positions))

    try:
        response = requests.get("https://api.openalex.org/works", params=params, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error retrieving data from OpenAlex: {e}")
        return []

    search_docs = []
    for result in response.json().get("results", []):
        primary = result.get("primary_location") or {}
        source = primary.get("source") or {}
        open_access = result.get("open_access") or {}
        search_docs.append({
            "title": result.get("title") or "Unknown Title",
            "authors": ", ".join(
                (auth.get("author") or {}).get("display_name", "")
                for auth in result.get("authorships", [])
            ),
            "year": result.get("publication_year"),
            "abstract": rebuild_abstract(result.get("abstract_inverted_index")),
            "doi": result.get("doi") or "",
            "url": primary.get("landing_page_url") or result.get("doi") or result.get("id", ""),
            "oa_url": open_access.get("oa_url") or "",
            "source": source.get("display_name") or "",
            "type": result.get("type") or "",
            "language": result.get("language") or "",
            "cited_by_count": result.get("cited_by_count", 0),
            "openalex_id": result.get("id", ""),
        })

    return search_docs

def search_wikipedia(query: str, max_results: int = 10):
    """Retrieve Wikipedia documents based on a query."""
    from langchain_community.document_loaders import WikipediaLoader
    search_docs = WikipediaLoader(query=query, load_max_docs=max_results).load()
    
    formatted_docs = []
    for doc in search_docs:
        formatted_docs.append({
            "title": doc.metadata.get("title", "Unknown Title"),
            "source": doc.metadata.get("source", "Unknown Source"),
            "page": doc.metadata.get("page", "Unknown Page"),
            "content": doc.page_content
        })
    
    return formatted_docs

# Example usage:
# if __name__ == "__main__":
#     topic = "machine learning"
#     articles = search_arxiv_sync(topic, fetch_full_paper=True)  # Set fetch_full_paper=True to download full paper

#     # Displaying the fetched results
#     for idx, article in enumerate(articles, 1):
#         print("-" * 50)
#         print(f"Paper {idx}:")
#         print(f"Title: {article['title']}")
#         print(f"ar5iv Link: {article['ar5iv_link']}")
#         print("-" * 20)
#         print(f"Abstract: {article['abstract']}...")  # Print part of the abstract
#         print("-" * 20)
#         print(f"HTML Content (first 300 chars): {article['full_content'][:3000]}...")
#         print("-" * 50)
#         print()
