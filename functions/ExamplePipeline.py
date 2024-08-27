from typing import List, Dict, Union
import json
import os
import asyncio
from neo4j import GraphDatabase
from pydantic import BaseModel
class CorrectiveActionPlanPipeline:
    class Valves(BaseModel):
        NEO4J_URI: str
        NEO4J_USER: str
        NEO4J_PASSWORD: str

    def __init__(self):
        self.valves = self.Valves(
            **{ 
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
                "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
            }
        )

    def neo4j_connection(self):
        return GraphDatabase.driver(self.valves.NEO4J_URI, auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD))

    def retrieve_and_visualize_similar_anomalies(self, problem: str, description: str, category: str, severity: str) -> List[Dict[str, str]]:
        # Retrieve similar anomalies based on provided parameters
        query = """
        MATCH (fa:FicheAnomalie)
        WHERE fa.fan_categorie = $category AND fa.fan_gravite_decision = $severity
        RETURN fa.fan_intitule AS title, fa.fan_description_anomalie AS description
        LIMIT 10
        """
        with self.neo4j_connection().session() as session:
            result = session.run(query, category=category, severity=severity)
            return [{"title": record["title"], "description": record["description"]} for record in result]

    def analyze_resolutions(self, similar_anomalies: List[Dict[str, str]]) -> List[str]:
        # Analyze resolutions based on similar anomalies
        resolutions = []
        for anomaly in similar_anomalies:
            resolutions.append(f"Resolution for '{anomaly['title']}': Implement fix based on findings.")
        return resolutions

    def create_corrective_action_plan(self, problem: str, description: str, category: str, severity: str) -> str:
        # Step A: Retrieve similar anomalies
        similar_anomalies = self.retrieve_and_visualize_similar_anomalies(problem, description, category, severity)

        if not similar_anomalies:
            return json.dumps({"error": "No similar anomalies found."})

        # Step B: Analyze resolutions from similar anomalies
        resolutions = self.analyze_resolutions(similar_anomalies)

        # Step C: Develop a structured corrective action plan
        corrective_action_plan = {
            "problem": problem,
            "description": description,
            "category": category,
            "severity": severity,
            "actions": resolutions,
            "responsible_persons": ["Assign responsible persons here"],  # To be filled
            "deadlines": ["Define deadlines here"],  # To be filled
        }

        # Step D: Return the plan in structured JSON format
        return json.dumps(corrective_action_plan)

    def pipe(self, problem: str, description: str, category: str, severity: str) -> str:
        return self.create_corrective_action_plan(problem, description, category, severity)
class ExamplePipeline:
    @staticmethod
    async def run_pipeline(problem: str, description: str, category: str, severity: str) -> str:
        pipeline = CorrectiveActionPlanPipeline()
        result = pipeline.pipe(problem, description, category, severity)
        return result
def test_pipeline():
    pipeline = ExamplePipeline()
    result = asyncio.run(pipeline.run_pipeline(
        problem="Managed Repositories and Proxied Repositories buttons under Administration are not displayed when using Internet Explorer 7.",
        description="Managed Repositories issue",
        category="Bug",
        severity="Major"
    ))
    assert result is not None, "The result must not be None"
    assert isinstance(result, str), "The result must be a string"

    result_data = json.loads(result)
    assert "problem" in result_data, "The 'problem' field must be in the result"
    assert "actions" in result_data, "The 'actions' field must be in the result"
    print("Tests passed.")
    return result_data