from typing import List, Dict, Union
import json
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
import os
import re
import unittest
import asyncio

class CorrectiveActionPlanPipeline:
    class Valves(BaseModel):
        NEO4J_URI: str
        NEO4J_USER: str
        NEO4J_PASSWORD: str

    def __init__(self):
        self.reset_pipeline()
        self.valves = self.Valves(
            **{ 
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
                "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
            }
        )

    def reset_pipeline(self):
        self.anomaly_data = {}
        self.similar_anomalies = []
        self.recommendations = []

    def neo4j_connection(self):
        return GraphDatabase.driver(self.valves.NEO4J_URI, auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD))

    def get_similar_anomalies(self, problem_title: str, problem_description: str):
        query = """
        MATCH (fa:FicheAnomalie)
        WHERE fa.fan_description_anomalie CONTAINS $description OR fa.fan_intitule CONTAINS $title
        RETURN fa.fan_description_anomalie AS description, fa.fan_intitule AS intitule
        LIMIT 10
        """
        with self.neo4j_connection().session() as session:
            result = session.run(query, title=problem_title, description=problem_description)
            return [{"title": record["intitule"], "description": record["description"]} for record in result]

    def analyze_resolutions(self, anomalies: List[Dict[str, str]]):
        # Mockup function to analyze resolutions
        # In a real scenario, this would likely pull from some data source to analyze resolutions
        return [{"resolution": "Review the patch logs and ensure proper deployment.", "responsible": "dev_team", "deadline": "2023-12-01"} for _ in anomalies]

    def create_corrective_action_plan(self, problem_title: str, problem_description: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        # Step 1: Retrieve similar anomalies
        self.similar_anomalies = self.get_similar_anomalies(problem_title, problem_description)

        # Step 2: Analyze resolutions for the similar anomalies
        self.recommendations = self.analyze_resolutions(self.similar_anomalies)

        # Step 3: Construct the corrective action plan
        plan = {
            "problem": {
                "title": problem_title,
                "description": problem_description
            },
            "similar_anomalies": self.similar_anomalies,
            "resolutions": self.recommendations
        }

        return plan

    def pipe(self, problem_title: str, problem_description: str) -> str:
        plan = self.create_corrective_action_plan(problem_title, problem_description)
        return json.dumps(plan, indent=4)

class ExamplePipeline:
    @staticmethod
    async def run_pipeline(problem_title: str, problem_description: str) -> str:
        pipeline = CorrectiveActionPlanPipeline()
        pipe_result = pipeline.pipe(problem_title, problem_description)
        return pipe_result



def test_pipeline():
    pipeline = ExamplePipeline()

    result = asyncio.run(pipeline.run_pipeline(
        "Managed Repositories issue", 
        "Managed Repositories and Proxied Repositories buttons under Administration are not displayed when using Internet Explorer 7."
    ))
    assert result is not None, "Le résultat ne doit pas être None"
    assert isinstance(result, str), "Le résultat doit être une chaîne de caractères"

    result_data = json.loads(result)
    assert "problem" in result_data, "Le champ 'problem' doit être dans le résultat"
    assert "similar_anomalies" in result_data, "Le champ 'similar_anomalies' doit être dans le résultat"
    assert "resolutions" in result_data, "Le champ 'resolutions' doit être dans le résultat"
    print("Tous les tests ont réussi !")
    return result_data

test_pipeline()