def analyse_jira_anomalies(problem, context="Critical"):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from neo4j import GraphDatabase
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # Connect to the Neo4j graph database
    URI = "bolt://127.0.0.1:7687"
    AUTH = ("neo4j", "password")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    # Query to retrieve anomaly details based on the specified filtering criteria
    query = f"""
    MATCH (fa:FicheAnomalie)
    RETURN fa.fan_description_anomalie AS description, fa.fan_intitule AS intitule, fa.fan_intitule AS title
    """
    
    # Retrieve anomaly data from the graph database
    with driver.session() as session:
        result = session.run(query)
        data = [{"title": record["intitule"], "description": record["description"]} for record in result]
    
    # Function to clean the text data
    def clean_text(data):
        return [item for item in data if isinstance(item['description'], str) and isinstance(item['title'], str)]
    
    # Function to compute embeddings for combined texts
    def compute_embeddings_for_combined_texts(texts):
        if not texts:
            raise ValueError("No texts provided for embedding computation.")
        
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Dimensionality reduction with TruncatedSVD
        n_components = min(100, X.shape[1])
        svd = TruncatedSVD(n_components=n_components)
        reduced_X = svd.fit_transform(X)
        
        return reduced_X, vectorizer, svd

    # Function to recalculate embeddings for test anomalies
    def recalculate_embeddings_for_test_anomalies(test_anomaly_title, test_anomaly_description, vectorizer, svd):
        combined_text = [f"{test_anomaly_title} {test_anomaly_description}"]
        X_test = vectorizer.transform(combined_text)
        return svd.transform(X_test)

    # Function to validate embeddings
    def validate_embeddings(embeddings):
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain NaN or infinite values.")

    # Function to get the most similar anomaly
    def get_most_similar_anomaly(test_anomaly_embedding, anomaly_embeddings):
        similarities = cosine_similarity(test_anomaly_embedding, anomaly_embeddings)
        most_similar_idx = np.argmax(similarities[0])
        most_similar_score = similarities[0][most_similar_idx]
        return most_similar_idx, most_similar_score

    # Clean the anomaly data
    cleaned_anomaly_data = clean_text(data)
    if not cleaned_anomaly_data:
        raise ValueError("No valid anomaly data found after cleaning.")
    
    # Create combined texts
    combined_texts = [f"{item['title']} {item['description']}" for item in cleaned_anomaly_data]
    anomaly_titles = [item['title'] for item in cleaned_anomaly_data]
    anomaly_abstracts = [item['description'] for item in cleaned_anomaly_data]
    
    if not combined_texts:
        raise ValueError("No combined texts available for embedding computation.")
    
    # Compute embeddings for combined texts
    anomaly_embeddings, vectorizer, svd = compute_embeddings_for_combined_texts(combined_texts)
    
    # Recalculate embeddings for the test anomaly
    test_anomaly_title = problem['title']
    test_anomaly_description = problem['abstract']
    test_anomaly_embeddings = recalculate_embeddings_for_test_anomalies(test_anomaly_title, test_anomaly_description, vectorizer, svd)
    
    # Validate embeddings
    validate_embeddings(test_anomaly_embeddings)
    validate_embeddings(anomaly_embeddings)
    
    # Get the most similar anomaly
    most_similar_idx, most_similar_score = get_most_similar_anomaly(test_anomaly_embeddings, anomaly_embeddings)
    
    # Return the most similar anomaly
    most_similar_anomaly = {
        "Most Similar Anomaly": {
            "Title": anomaly_titles[most_similar_idx],
            "Description": anomaly_abstracts[most_similar_idx],
            "Similarity": f"{most_similar_score:.4f}"
        }
    }
    
    return most_similar_anomaly

import json
# Example usage
problem = {
    'title': 'Sample Anomaly Title',
    'abstract': 'Sample anomaly description for testing purposes.'
}
result = analyse_jira_anomalies(problem)
print(json.dumps(result, indent=2))
