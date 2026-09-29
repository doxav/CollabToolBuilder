def search_hal(query: str, max_results: int = 20, year_from: int = None, year_to: int = None,
               doc_types: list = None, language: str = None, with_fulltext_only: bool = False):
    """Search the French open archive HAL (hal.science) for scientific documents.

    Args:
        query: free-text query (French or English), e.g. "réseaux sociaux violence".
        max_results: maximum number of documents to return (HAL allows up to 10000).
        year_from: keep documents produced from this year (e.g. 2022), inclusive.
        year_to: keep documents produced up to this year, inclusive.
        doc_types: HAL document types to keep, e.g. ["ART", "COMM", "THESE", "OUV", "COUV"].
        language: ISO code of the document language, e.g. "fr" or "en".
        with_fulltext_only: if True, keep only documents with an attached PDF.

    Returns:
        A list of dicts with keys: hal_id, title, authors, year, doc_type, language,
        abstract, keywords, url, pdf_url.
    """
    import requests

    fq = []
    if year_from or year_to:
        fq.append(f"producedDateY_i:[{year_from or '*'} TO {year_to or '*'}]")
    if doc_types:
        fq.append("docType_s:(" + " OR ".join(doc_types) + ")")
    if language:
        fq.append(f"language_s:{language}")
    if with_fulltext_only:
        fq.append("submitType_s:file")

    params = {
        "q": query,
        "fq": fq,
        "rows": max_results,
        "wt": "json",
        "sort": "score desc",
        "fl": ",".join([
            "halId_s", "title_s", "authFullName_s", "producedDateY_i", "docType_s",
            "language_s", "abstract_s", "keyword_s", "uri_s", "fileMain_s",
        ]),
    }

    try:
        response = requests.get("https://api.archives-ouvertes.fr/search/", params=params, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error retrieving data from HAL: {e}")
        return []

    def first(value, default=""):
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    results = []
    for doc in response.json().get("response", {}).get("docs", []):
        results.append({
            "hal_id": doc.get("halId_s", ""),
            "title": first(doc.get("title_s"), "Unknown Title"),
            "authors": ", ".join(doc.get("authFullName_s", [])),
            "year": doc.get("producedDateY_i"),
            "doc_type": doc.get("docType_s", ""),
            "language": first(doc.get("language_s")),
            "abstract": first(doc.get("abstract_s"), "No abstract available"),
            "keywords": doc.get("keyword_s", []),
            "url": doc.get("uri_s", ""),
            "pdf_url": doc.get("fileMain_s", ""),
        })
    return results
