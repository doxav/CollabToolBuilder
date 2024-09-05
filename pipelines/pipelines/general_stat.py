from typing import List, Union, Generator, Iterator
import json
import numpy as np
from neo4j import GraphDatabase
from pydantic import BaseModel
from collections import Counter
import openai   
import os
import re



class Pipeline:
    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        NEO4J_URI: str
        NEO4J_USER: str
        NEO4J_PASSWORD: str
        OPENAI_API_KEY: str

    def __init__(self):
        self.reset_pipeline()

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3_8b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            }
        )

        openai.api_key = self.valves.OPENAI_API_KEY

    
    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        global documents, index

        # Optionally, load documents if needed for LlamaIndex
        #self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        #self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # Actions to perform on server shutdown
        pass

    def reset_pipeline(self):
        self.statistics = {
            "most_common_anomalies": None,
            "users_with_most_reports": None,
            "programs_with_most_issues": None
        }

    def neo4j_connection(self):
        return GraphDatabase.driver(self.valves.NEO4J_URI, auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD))

    def get_anomalies_data(self, driver):
        query = """
        MATCH (fa:FicheAnomalie)
        RETURN fa.fan_intitule AS title, fa.fan_description_anomalie AS description, 
               fa.fan_responsable_declaration AS reporter, fa.fan_programme AS program
        """
        with driver.session() as session:
            result = session.run(query)
            return [
                {
                    "title": record["title"],
                    "description": record["description"],
                    "reporter": record["reporter"],
                    "program": record["program"],
                }
                for record in result
            ]

    def analyze_data(self, data):
        titles = [item['title'] for item in data if item['title']]
        reporters = [item['reporter'] for item in data if item['reporter']]
        programs = [item['program'] for item in data if item['program']]

        most_common_anomalies = Counter(titles).most_common(5)
        users_with_most_reports = Counter(reporters).most_common(5)
        programs_with_most_issues = Counter(programs).most_common(5)

        self.statistics['most_common_anomalies'] = most_common_anomalies
        self.statistics['users_with_most_reports'] = users_with_most_reports
        self.statistics['programs_with_most_issues'] = programs_with_most_issues

    def generate_statistics_report(self):
        report = "📊 **Statistical Analysis of Anomalies** 📊\n\n"
        
        report += "🔍 **Most Common Anomalies** 🔍\n"
        for idx, (title, count) in enumerate(self.statistics['most_common_anomalies'], 1):
            report += f"   {idx}. **Title**: {title} - **Occurrences**: {count}\n"

        report += "\n👥 **Top Users Reporting Anomalies** 👥\n"
        for idx, (user, count) in enumerate(self.statistics['users_with_most_reports'], 1):
            report += f"   {idx}. **User**: {user} - **Reports**: {count}\n"

        report += "\n💻 **Programs with Most Issues** 💻\n"
        for idx, (program, count) in enumerate(self.statistics['programs_with_most_issues'], 1):
            report += f"   {idx}. **Program**: {program} - **Issues**: {count}\n"

        return report

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
            ) -> Union[str, Generator, Iterator]:

        driver = self.neo4j_connection()
        try:
            data = self.get_anomalies_data(driver)
        finally:
            driver.close()

        if not data:
            return "No data found in the database."

        self.analyze_data(data)
        report = self.generate_statistics_report()
        
        self.reset_pipeline()

        return report


class AnomalyStatisticalAnalysisPipeline:
    @staticmethod
    async def run_pipeline(user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result =  pipeline.pipe(user_message, model_id, messages, body)
        
        await pipeline.on_shutdown()
        return pipe_result