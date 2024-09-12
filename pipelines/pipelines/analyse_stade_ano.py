from typing import List, Dict, Union
import numpy as np
from neo4j import GraphDatabase
import os
import re
import matplotlib.pyplot as plt
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        NEO4J_URI: str
        NEO4J_USER: str
        NEO4J_PASSWORD: str

    def __init__(self):
        self.reset_pipeline()
        self.valves = self.Valves(
            **{ 
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
            }
        )

    def reset_pipeline(self):
        self.phase_data = {}
        self.conversation_state = "start"

    async def on_startup(self):
        pass
        
    async def on_shutdown(self):
        pass

    def neo4j_connection(self):
        return GraphDatabase.driver(self.valves.NEO4J_URI, auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD))

    def get_anomalies_by_phase(self, driver):
        query = """
        MATCH (fa:FicheAnomalie)
        RETURN fa.fan_stade_generateur_anomalie AS phase, COUNT(*) AS nombre_anomalies
        ORDER BY nombre_anomalies DESC
        """
        with driver.session() as session:
            result = session.run(query)
            return [{"phase": record["phase"], "count": record["nombre_anomalies"]} for record in result]

    def aggregate_phase_data(self, data):
        return {record['phase']: record['count'] for record in data}

    def generate_graph(self, phase_data):
        phases = list(phase_data.keys())
        counts = list(phase_data.values())

        plt.figure(figsize=(10, 6))
        plt.barh(phases, counts, color='skyblue')
        plt.xlabel('Nombre d\'anomalies')
        plt.ylabel('Phase génératrice d\'anomalies')
        plt.title('Répartition des anomalies par phase génératrice')
        plt.tight_layout()

        # Sauvegarder le graphique sous forme d'image
        image_path = "anomalies_par_phase.png"
        plt.savefig(image_path)

        return image_path

    def generate_recommendations(self, phase_data):
        recommendations = {
            "Développement": "Augmenter les revues de code pour éviter les bugs en développement.",
            "Maintenance": "Renforcer les tests après chaque mise à jour.",
            "Test": "Augmenter la fréquence des tests de régression."
        }
        top_phases = sorted(phase_data.items(), key=lambda x: x[1], reverse=True)[:3]
        recommendation_text = "Recommandations pour les phases génératrices d'anomalies :\n"
        
        for phase, count in top_phases:
            recommendation_text += f"Phase: {phase} - {count} anomalies\n"
            recommendation_text += f"Recommandation: {recommendations.get(phase, 'Aucune recommandation spécifique.')}\n\n"
        
        return recommendation_text

    def process_anomaly_phases(self):
        driver = self.neo4j_connection()
        try:
            # Step 1: Get data
            data = self.get_anomalies_by_phase(driver)
        finally:
            driver.close()

        # Step 2: Aggregate and Analyze
        phase_data = self.aggregate_phase_data(data)

        # Step 3: Generate Graph
        image_path = self.generate_graph(phase_data)

        # Step 4: Generate recommendations
        recommendations = self.generate_recommendations(phase_data)

        return recommendations, image_path

    def pipe(self, user_message: str, model_id: str, messages: List[Dict], body: Dict) -> Union[str, None]:
        recommendations, image_path = self.process_anomaly_phases()
        
        result = (
            "🛠 **Analyse des phases génératrices d'anomalies** 🛠\n\n"
            f"{recommendations}\n"
            f"Le graphique des anomalies par phase a été généré : {image_path}\n"
        )

        return result

class PhaseAnalysisPipeline:
    @staticmethod
    async def run_pipeline(user_message: str) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message)
        await pipeline.on_shutdown()
        return pipe_result
