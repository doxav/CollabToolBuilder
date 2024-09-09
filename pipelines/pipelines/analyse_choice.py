from typing import List, Union
import os
from neo4j import GraphDatabase
from pydantic import BaseModel
from collections import Counter
import ipywidgets as widgets
from IPython.display import Markdown    
import pandas as pd


class Pipeline:
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

    def get_anomalies_data(self, driver):
        query = """
        MATCH (fa:FicheAnomalie)
        RETURN fa.fan_intitule AS title, fa.fan_description_anomalie AS description, 
               fa.fan_responsable_declaration AS reporter, fa.fan_programme AS program,
               fa.fan_batiment AS building, fa.fan_gravite_decision AS severity,
               fa.fan_stade_generateur_anomalie AS stage
        """

        with driver.session() as session:
            result = session.run(query)
            data = [
                {
                    "title": record["title"],
                    "description": record["description"],
                    "reporter": record["reporter"],
                    "program": record["program"],
                    "building": record["building"],
                    "severity": record["severity"],
                    "stage": record["stage"],
                }
                for record in result
            ]

        return data

    def analyze_data(self, data, choice: str):
        if choice == "1":
            titles = [item['title'] for item in data if item['title']]
            return Counter(titles).most_common(5)

        elif choice == "2":
            reporters = [item['reporter'] for item in data if item['reporter']]
            return Counter(reporters).most_common(5)

        elif choice == "3":
            programs = [item['program'] for item in data if item['program']]
            return Counter(programs).most_common(5)

        elif choice == "4":
            buildings = [item['building'] for item in data if item['building']]
            return Counter(buildings).most_common(5)

        elif choice == "5":
            severities = [item['severity'] for item in data if item['severity']]
            return Counter(severities).most_common(5)

        elif choice == "6":
            stages = [item['stage'] for item in data if item['stage']]
            return Counter(stages).most_common(5)

        return None

    def generate_statistics_report(self, analysis_results, choice: str):

        if not analysis_results:
            print("No data found or incorrect option selected.")
            return

        # Define the headers based on the user's choice
        headers = {
            "1": ("🔍 **Most Common Anomalies** 🔍", "Occurrences"),
            "2": ("👥 **Top Users Reporting Anomalies** 👥", "Reports"),
            "3": ("💻 **Programs with Most Issues** 💻", "Issues"),
            "4": ("🏢 **Most Affected Buildings*** 🏢", "Occurrences"),
            "5": ("⚠️ **Severity Levels** ⚠️", "Occurrences"),
            "6": ("🔧 **Most Generator of Anomalies** 🔧", "Occurrences"),
        }

        header_name, header_count = headers[choice]

        # Create a DataFrame from the analysis results
        df = pd.DataFrame(analysis_results, columns=[header_name, header_count])

        # Convert the styled DataFrame to HTML
        html_table = df.to_markdown()
        
        # Display the widget
        return html_table  


    def display_menu(self) -> str:
        menu = """
        ## 📋 Please Select an Option:
        
        1. **View Most Common Anomalies** 🔍
        2. **View Top Users Reporting Anomalies** 👥
        3. **View Programs with Most Issues** 💻
        4. **View Most Affected Buildings** 🏢
        5. **View Most Severe Anomalies** ⚠️
        6. **View Most Active Stages** 🔧
        7. **The End** 🏁
        """
        return menu

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        choice = user_message
        menu = self.display_menu()
        if choice not in [str(i) for i in range(1, 8)]:
            # Invalid choice or first interaction, show menu
            return self.display_menu()

        if choice == "7":
            return "✅ **The End** ✅\nThank you for using the anomaly analysis tool."

        driver = self.neo4j_connection()
        try:
            data = self.get_anomalies_data(driver)
        finally:
            driver.close()

        if not data:
            return f"{'No data found in the database', menu}"
        
        analysis_results = self.analyze_data(data, choice)
        report = self.generate_statistics_report(analysis_results, choice)
        report += self.display_menu()

        return report

class AnomalyStatisticalAnalysisPipeline:
    @staticmethod
    async def run_pipeline(user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message, model_id, messages, body)
        await pipeline.on_shutdown()
        return pipe_result
