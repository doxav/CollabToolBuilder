from env.IR_CPS_TechSynthesis.env import SynthesisManager, Section

# This is the main function that will be called by the CPS
# It will return a list of articles given a query
def searchTop25ResearchPaper(bot: SynthesisManager, query, MAX_ARTICLES=25, MAX_SOURCES=10, SIMILARITY_THRESHOLD = 0.85, search_sources = ['paper_arxiv', 'websearch_wikipedia','paper_semantic_scholar']):
    import requests
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    def compute_embeddings(texts):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return model.encode(texts)

    def remove_similar_articles(articles, SIMILARITY_THRESHOLD):
        titles = [article['title'] for article in articles]
        embeddings = compute_embeddings(titles)
        similarity_matrix = cosine_similarity(embeddings)

        to_remove = set()
        for i in range(len(titles)):
            for j in range(i+1, len(titles)):
                if similarity_matrix[i][j] > SIMILARITY_THRESHOLD:
                    to_remove.add(j)

        unique_articles = [article for idx, article in enumerate(articles) if idx not in to_remove]
        return unique_articles

    # Collect the results from each source
    all_results = []
    for source in search_sources:
        # search_generic returns a list of result dict, each of format: {'title': '...', 'link': '...', 'description': '...'}
        results = SynthesisManager.search_generic(query, search_type=source, max_results=MAX_ARTICLES)
        all_results.extend(results)

    unique_results = remove_similar_articles(all_results, SIMILARITY_THRESHOLD)[:MAX_ARTICLES]

    bot.add_or_update_results_in_resources(unique_results)

    return unique_results
