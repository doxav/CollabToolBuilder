def retrieve_and_visualize_similar_anomalies(problem):
    """
    The function retrieves and visualizes anomalies similar to a given problem by querying a Neo4j database, processing the data,
    and calculating cosine similarities between the provided problem and the retrieved anomalies. It returns a JSON object 
    containing the original problem and a list of 3 similar anomalies. The function manages database connections and handles 
    data cleaning and embedding calculations internally. Finally, it filters and ranks the most similar anomalies based on 
    their textual descriptions.
    
    :param problem: A dictionary containing the title and abstract of the anomaly to compare against
    :return: A JSON string containing the problem, a list of 3 similar anomalies and a recommendation text depending on the similarities
    """
    from neo4j import GraphDatabase
    import json
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import openai
    import os
    from langchain_openai import OpenAI

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Step 1: Connect to Neo4j database
    def neo4j_connection():
        URI = "bolt://127.0.0.1:7687"
        AUTH = ("neo4j", "password")
        return GraphDatabase.driver(URI, auth=AUTH)

    # Step 2: Retrieve anomalies from the database
    def get_anomalies(driver):
        query = """
        MATCH (fa:FicheAnomalie)
        RETURN fa.fan_description_anomalie AS description, fa.fan_intitule AS intitule
        """
        with driver.session() as session:
            result = session.run(query)
            return [{"title": record["intitule"], "description": record["description"]} for record in result]

    # Step 3: Clean the retrieved anomaly data
    def clean_text(data):
        return [item for item in data if isinstance(item['description'], str) and isinstance(item['title'], str)]

    # Step 4: Compute embeddings for the combined texts
    def compute_embeddings_for_combined_texts(texts):
        if not texts:
            raise ValueError("No texts provided for embedding computation.")
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        n_components = min(100, X.shape[1])
        svd = TruncatedSVD(n_components=n_components)
        reduced_X = svd.fit_transform(X)
        return reduced_X, vectorizer, svd

    # Step 5: Recalculate embeddings for the new anomaly
    def recalculate_embeddings_for_test_anomalies(test_anomaly_title, test_anomaly_description, vectorizer, svd):
        combined_text = [f"{test_anomaly_title} {test_anomaly_description}"]
        X_test = vectorizer.transform(combined_text)
        return svd.transform(X_test)

    # Step 6: Validate embeddings for NaN or infinite values
    def validate_embeddings(embeddings):
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain NaN or infinite values.")

    # Step 7: Retrieve similar anomalies based on cosine similarity
    def get_most_similar_anomalies(test_anomaly_embedding, anomaly_embeddings):
        similarities = cosine_similarity(test_anomaly_embedding, anomaly_embeddings)
        most_similar_indices = np.argsort(-similarities[0])[:3]  # Get top 3 similar
        return most_similar_indices, similarities[0]
    
    def generate_recommendation_text(similar_anomalies, problem):
        prompt = f"The user has encountered a problem described as follows:\n\nTitle: {problem['title']}\nDescription: {problem['abstract']}\n\nBased on this problem, here are descriptions of 3 similar anomalies:\n\n"
        for idx, anomaly in enumerate(similar_anomalies):
            prompt += f"Anomaly {idx + 1}:\nTitle: {anomaly['title']}\nDescription: {anomaly['description']}\n\n"
        prompt += "Please provide a detailed recommendation to solve the user's problem based on the similarities with these anomalies."

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at providing solutions for software anomalies."},
                {"role": "user", "content": prompt}
            ]
        )

        # Access the content correctly
        recommendation_text = response.choices[0].message.content
        return recommendation_text.strip()


    
    # Step 9: Write the result to a file
    def write_to_file(result, filename="result_functions.txt"):
        with open(filename, "w") as file:
            file.write(result)

    # Main execution flow
    driver = neo4j_connection()
    try:
        data = get_anomalies(driver)
    finally:
        driver.close()

    cleaned_anomaly_data = clean_text(data)
    if not cleaned_anomaly_data:
        raise ValueError("No valid anomaly data found after cleaning.")

    combined_texts = [f"{item['title']} {item['description']}" for item in cleaned_anomaly_data]
    anomaly_embeddings, vectorizer, svd = compute_embeddings_for_combined_texts(combined_texts)

    test_anomaly_title = problem['title']
    test_anomaly_description = problem['abstract']
    test_anomaly_embeddings = recalculate_embeddings_for_test_anomalies(test_anomaly_title, test_anomaly_description, vectorizer, svd)

    validate_embeddings(test_anomaly_embeddings)
    validate_embeddings(anomaly_embeddings)

    most_similar_indices, similarity_scores = get_most_similar_anomalies(test_anomaly_embeddings, anomaly_embeddings)

    top3_anomalies = [
        {
            "title": cleaned_anomaly_data[idx]["title"],
            "description": cleaned_anomaly_data[idx]["description"],
            "similarity_score": similarity_scores[idx],
        }
        for idx in most_similar_indices
    ]

    recommendation_text = generate_recommendation_text(top3_anomalies, problem)

    result = json.dumps({
        "problem": problem,
        "top 3 anomalies": top3_anomalies,
        "recommendation_text": recommendation_text
    }, indent=4)
    
    write_to_file(result)

    return result

# Example usage
problem = {
    'title': 'Internal server connection error',
    'abstract': 'A server encountered an internal error and could not complete your request.'
}

test = retrieve_and_visualize_similar_anomalies(problem)

print(test)
