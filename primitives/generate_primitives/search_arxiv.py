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
