from typing import List
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    def __init__(self, min_cosine_similarity: float, target_plan_embedding: List[float]):
        self.min_cosine_similarity = min_cosine_similarity
        self.target_plan_embedding = target_plan_embedding

    def normalized_cosine_similarity(self, a: List[float], b: List[float], min_cs: float = None) -> float:
        if min_cs is None:
            min_cs = self.min_cosine_similarity
        return (cosine_similarity([a], [b])[0][0] - min_cs) / (1 - min_cs)

    def calculate_similarity(self, plan_embedding: List[float]):
        plan_embedding_similarity = self.normalized_cosine_similarity(
            plan_embedding, self.target_plan_embedding, self.min_cosine_similarity
        )
        print("self.min_plan_cosine_similarity", self.min_cosine_similarity)
        print("plan_embedding_similarity", plan_embedding_similarity)
        return plan_embedding_similarity

# Example usage:
target_embedding = [0.1, 0.2, 0.3, 0.4]
plan_embedding = [0.2, 0.1, 0.4, 0.3]
min_cos_sim = 0.5

calculator = SimilarityCalculator(min_cosine_similarity=min_cos_sim, target_plan_embedding=target_embedding)
similarity = calculator.calculate_similarity(plan_embedding)
