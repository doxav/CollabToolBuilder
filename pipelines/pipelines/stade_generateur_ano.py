from typing import List, Dict, Union
import numpy as np
from neo4j import GraphDatabase
import os
import matplotlib
import re
matplotlib.use('Agg')  # Use a non-interactive backend
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
   
    @staticmethod
    def sanitize_for_plotting(text):
        # Replace problematic characters for LaTeX
        # Escape special characters
        sanitized_text = text.replace('$', r'\$').replace('_', r'\_').replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}')
        # Remove problematic sequences
        sanitized_text = re.sub(r'[^a-zA-Z0-9\s\$\_\{\}\\]', '', sanitized_text)
        return sanitized_text

    def aggregate_phase_data(self, data):
        return {record['phase']: record['count'] for record in data}

    def generate_graph(self, phase_data):
        filtered_phase_data = {phase: count for phase, count in phase_data.items() if phase is not None and count is not None}

        phases = list(filtered_phase_data.keys())
        counts = list(filtered_phase_data.values())

        if not phases or not counts:
            return "No valid data to plot."

        # Get top 5 phases
        top_phases = sorted(filtered_phase_data.items(), key=lambda x: x[1], reverse=True)[:5]
        phases, counts = zip(*top_phases)

        plt.figure(figsize=(10, 6))
        plt.barh(phases, counts, color='skyblue')
        plt.xlabel(self.sanitize_for_plotting("Nombre d anomalies"))
        plt.ylabel(self.sanitize_for_plotting("Phase génératrice d anomalies"))
        plt.title(self.sanitize_for_plotting("Répartition des anomalies par phase génératrice"))
        plt.rcParams['text.usetex'] = False
        plt.tight_layout()

        image_path = 'graph_from_pipelines/graph_stade_generateur.png'
        plt.savefig(image_path)
        plt.close()  # Close the plot to free up resources
        return image_path

    def generate_recommendations(self, phase_data):
        
        # Get top 5 phases
        top_phases = sorted(phase_data.items(), key=lambda x: x[1], reverse=True)[:5]
        recommendation_text = "**Top des phases les plus génératrices d'anomalies :**\n"
        
        for phase, count in top_phases:
            recommendation_text += f"  -  Phase de production: **{phase}** - {count} anomalies\n\n\n"
            
        
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
