import requests
import os

paper_id = "ARXIV:2212.06094"  # LMQL: Language Model Query Language
paper_id = "ARXIV:2305.13971" # Grammar-constrained Decoding for structured NLP

def fetch_citations(paper_id, offset=0, limit=100, fields='paperId,title,openAccessPdf'):
    # Construct the URL for the Semantic Scholar API request
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=openAccessPdf,externalIds,title"
    print(f"API GET Citations URL: {url}")
    params = {
        'offset': offset,
        'limit': limit,
        'fields': fields
    }

    # Make the request to Semantic Scholar API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching citations: {response.text}")

def download_pdf_from_url(url, directory, file_name):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name.replace('/', '_').replace('\\', '_'))
    # check if file already exists, skip downloading
    if os.path.exists(file_path):
        print(f"File already exists: {file_name}")
        return
    # Make the request to download the PDF
    response = requests.get(url['url'])
    if response.status_code == 200:
        # Ensure the directory exists
        # Write the PDF to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download {file_name}")

def download_pdf_from_arxiv_id(arxiv_id, directory, name=''):
    if arxiv_id:
        paper_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        file_name = ((name+"_" if name else "") + f"{arxiv_id}.pdf").replace('/', '_').replace('\\', '_')
        file_path = os.path.join(directory, file_name)
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # if file already exists, skip downloading
        if os.path.exists(file_path):
            print(f"File already exists: {file_name}")
            return
        
        # Make the request to download the PDF
        response = requests.get(paper_url)
        if response.status_code == 200:
            # Write the PDF to a file
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {file_name}")
        else:
            print(f"Failed to download {file_name}")
    else:
        print("No ArXiv ID found for paper.")

# Example usage
if __name__ == "__main__":
    try:
        citations_data = fetch_citations(paper_id)
        cited_papers_dir = "cited_papers"  # Directory to store downloaded PDFs
        
        for citation in citations_data.get('data', []):
            citing_paper = citation.get('citingPaper', {})
            paper_title = citing_paper.get('title', 'Untitled').replace('/', '_').replace('\\', '_')[:50]  # Sanitize and truncate title
            openaccess_pdf_url = citing_paper.get('openAccessPdf')
            external_ids = citing_paper.get('externalIds', {})
            arxiv_id = external_ids.get('ArXiv')
            if arxiv_id:
                download_pdf_from_arxiv_id(arxiv_id, cited_papers_dir, citing_paper.get('title', ''))
            else:
                if openaccess_pdf_url:
                    file_name = paper_title + ("_"+arxiv_id if arxiv_id else "") + ".pdf"
                    download_pdf_from_url(openaccess_pdf_url, cited_papers_dir, file_name)
                else:
                    print(f"No ArXiv ID and OpenAccessPdf available for: {citing_paper.get('title', 'Untitled')} / citingPaper: {citing_paper}")
    except Exception as e:
        print(e)
