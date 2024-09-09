from typing import List, Dict, Union
import os
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pydantic import BaseModel
class AnomalyClusteringPipeline:
    class Valves(BaseModel):
        NEO4J_URI: str
        NEO4J_USER: str
        NEO4J_PASSWORD: str

    def __init__(self):
        self.valves = self.Valves(
            **{
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
            }
        )

    def neo4j_connection(self):
        return GraphDatabase.driver(self.valves.NEO4J_URI, auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD))

    def get_anomalies(self, driver):
        query = """
        MATCH (fa:FicheAnomalie)
        RETURN fa.fan_description_anomalie AS description, fa.fan_intitule AS intitule
        """
        with driver.session() as session:
            result = session.run(query)
            return [{"title": record["intitule"], "description": record["description"]} for record in result]

    def clean_text(self, data):
        return [item for item in data if isinstance(item['description'], str) and isinstance(item['title'], str)]

    def compute_embeddings_for_combined_texts(self, texts):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        n_components = min(100, X.shape[1])
        svd = TruncatedSVD(n_components=n_components)
        reduced_X = svd.fit_transform(X)
        return reduced_X, vectorizer, svd

    def recalculate_embeddings_for_test_anomalies(self, test_anomaly_title, test_anomaly_description, vectorizer, svd):
        combined_text = [f"{test_anomaly_title} {test_anomaly_description}"]
        X_test = vectorizer.transform(combined_text)
        return svd.transform(X_test)

    def get_most_similar_anomalies(self, test_anomaly_embedding, anomaly_embeddings):
        similarities = cosine_similarity(test_anomaly_embedding, anomaly_embeddings)
        most_similar_indices = np.argsort(-similarities[0])[:3]
        return most_similar_indices, similarities[0]

    def generate_report(self, top3_anomalies):
        report = "🔍 **Top 3 Similar Anomalies** 🔍\n"
        for i, anomaly in enumerate(top3_anomalies, 1):
            report += (
                f"   {i}. **Anomaly**\n"
                f"      - **Title**: {anomaly['title']}\n"
                f"      - **Description**: {anomaly['description']}\n"
            )
        return report

    def pipe(self, user_message: str) -> str:
        # Connect to Neo4j and retrieve anomalies
        driver = self.neo4j_connection()
        try:
            data = self.get_anomalies(driver)
        finally:
            driver.close()

        # Clean the data
        cleaned_anomaly_data = self.clean_text(data)
        if not cleaned_anomaly_data:
            return "No valid anomaly data found after cleaning!"

        combined_texts = [f"{item['title']} {item['description']}" for item in cleaned_anomaly_data]
        anomaly_embeddings, vectorizer, svd = self.compute_embeddings_for_combined_texts(combined_texts)

        # Assume user_message contains title and description of the new anomaly
        test_anomaly_title, test_anomaly_description = user_message.split("|")  # Example input
        test_anomaly_embedding = self.recalculate_embeddings_for_test_anomalies(
            test_anomaly_title.strip(), test_anomaly_description.strip(), vectorizer, svd
        )

        # Find most similar anomalies
        most_similar_indices, similarity_scores = self.get_most_similar_anomalies(test_anomaly_embedding, anomaly_embeddings)

        top3_anomalies = [
            {"title": cleaned_anomaly_data[idx]["title"], "description": cleaned_anomaly_data[idx]["description"]}
            for idx in most_similar_indices
        ]

        report = self.generate_report(top3_anomalies)
        assert report is not None, "Result can't be None"
        assert isinstance(report, str), "Result should be str"
        return report

class AnomalyClusteringAndRecommendationPipeline:
    @staticmethod
    async def run_pipeline(user_message: str) -> str:
        pipeline = AnomalyClusteringPipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message)
        await pipeline.on_shutdown()
        return pipe_result