from typing import List, Union, Generator, Iterator
import json
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import openai   
import os
import re
from pydantic import BaseModel
from dotenv import load_dotenv



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
        load_dotenv()
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
        

    def reset_pipeline(self):
        self.documents = None
        self.index = None
        self.anomaly_data = {}
        self.conversation_state = "start"

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

    def extract_title_from_prompt(self, user_input: str) -> str:
        # Utilise une expression régulière pour extraire la partie après "here is the title of my anomaly: "
        match = re.search(r"the title of my anomaly: (.+)", user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return None

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

    def generate_recommendation_text(self, similar_anomalies, problem):
        prompt = f"The user has encountered a problem described as follows:\n\nTitle: {problem['title']}\nDescription: {problem['abstract']}\n\nBased on this problem, here are descriptions of 3 similar anomalies:\n\n"
        for idx, anomaly in enumerate(similar_anomalies):
            prompt += f"Anomaly {idx + 1}:\nTitle: {anomaly['title']}\nDescription: {anomaly['description']}\n\n"
        prompt += "Please provide a detailed recommendation to solve the user's problem based on the similarities with these anomalies."
        
        openai.api_key = self.valves.OPENAI_API_KEY

        response = openai.chat.completions.create(
           
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at providing solutions for software anomalies."},
                {"role": "user", "content": prompt}
            ],
            

        )

        return response.choices[0].message.content.strip()
    

    def ask_next_question(self):
        if self.conversation_state == "start":
            return "Welcome to the anomaly reporting system. Please provide the title of the anomaly you encountered"
        elif self.conversation_state == "ask_title":
            return "Ok let's start ! Please provide the title of the anomaly."
        elif self.conversation_state == "ask_abstract":
            return "Please provide a brief abstract of the anomaly."
        elif self.conversation_state == "ask_number":
            return "Now, Please provide the anomaly number."
        elif self.conversation_state == "ask_comment":
            return "Please provide any additional comments about the anomaly."
        elif self.conversation_state == "confirmation":
            return f"Here is the summary of the anomaly:\n\nTitle: {self.anomaly_data['title']}\nAbstract: {self.anomaly_data['abstract']}\nNumber: {self.anomaly_data['number']}\nComment: {self.anomaly_data['comment']}\n\nIs this information correct? (yes/no)"
        else:
            return "Thanks for using our pipeline. Please enter a whitespace to start again."

    def process_user_response(self, user_input):

        new_title = self.extract_title_from_prompt(user_input)
        if new_title:
            self.anomaly_data['title'] = new_title
            self.conversation_state = "ask_title"  
            return self.ask_next_question()
        
        user_input = user_input.strip()

        if self.conversation_state == "start" and user_input != "reset":
            self.anomaly_data['start'] = user_input
            self.conversation_state = "first_query"
            return self.ask_next_question()
        
        elif self.conversation_state == "first_query" and user_input != "reset":
            self.anomaly_data['first_query'] = user_input
            self.conversation_state = "ask_title"
            return self.ask_next_question()
        
        elif self.conversation_state == "ask_title" and user_input != "reset":
            self.anomaly_data['title'] = user_input
            self.conversation_state = "ask_abstract"
            return self.ask_next_question()
        
        elif self.conversation_state == "ask_abstract" and user_input != "reset":
            self.anomaly_data['abstract'] = user_input
            self.conversation_state = "ask_number"
            return self.ask_next_question()
        
        elif self.conversation_state == "ask_number" and user_input != "reset":
            self.anomaly_data['number'] = user_input
            self.conversation_state = "ask_comment"
            return self.ask_next_question()
        
        elif self.conversation_state == "ask_comment" and user_input != "reset":
            self.anomaly_data['comment'] = user_input
            self.conversation_state = "confirmation"
            return self.ask_next_question()
        
        elif self.conversation_state == "confirmation" and user_input != "reset":
            if user_input.lower() in ["yes", "y"]:
                self.conversation_state = "finished"
                return "Thanks ! let's start processing the anomaly data. Please enter to proceed."
            else:
                self.conversation_state = "ask_title"
                self.anomaly_data = {}
                return "Oh no ! Let's start again. Please provide the title of the new anomaly."
            
        elif user_input == "reset":
            self.reset_pipeline()
            user_input = ""
            return "The conversation has been reset."

        else:
            # Réinitialiser la conversation si l'état est inconnu
            self.anomaly_data = {}
            self.conversation_state = "ask_title"
            return "Let's start again. Please provide the title of the new anomaly."


    def handle_conversation(self, user_input):
        return self.process_user_response(user_input)

    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        result = ""

        if self.conversation_state != "finished":
            return self.handle_conversation(user_message)

        # Assuming the user_message contains the problem description
        problem = {
            'title': self.anomaly_data['title'],
            'abstract': self.anomaly_data['abstract'],
            'number': self.anomaly_data['number'],
            'comment': self.anomaly_data['comment']
        }

        # Step 1: Connect to Neo4j and retrieve anomalies
        driver = self.neo4j_connection()
        try:
            data = self.get_anomalies(driver)
        finally:
            driver.close()

        # Step 2: Process and clean the data
        cleaned_anomaly_data = self.clean_text(data)
        if not cleaned_anomaly_data:
            return "No valid anomaly data found after cleaning."

        combined_texts = [f"{item['title']} {item['description']}" for item in cleaned_anomaly_data]
        anomaly_embeddings, vectorizer, svd = self.compute_embeddings_for_combined_texts(combined_texts)

        test_anomaly_embeddings = self.recalculate_embeddings_for_test_anomalies(
            problem['title'], problem['abstract'], vectorizer, svd
        )

        # Step 3: Find most similar anomalies
        most_similar_indices, similarity_scores = self.get_most_similar_anomalies(test_anomaly_embeddings, anomaly_embeddings)

        top3_anomalies = [
            {
                "title": cleaned_anomaly_data[idx]["title"],
                "description": cleaned_anomaly_data[idx]["description"],
                "similarity_score": similarity_scores[idx],
            }
            for idx in most_similar_indices
        ]

        openai.api_key = self.valves.OPENAI_API_KEY

        # Step 4: Generate recommendation
        recommendation_text = self.generate_recommendation_text(top3_anomalies, problem)
        
        # Step 5: Return the result
        result = (
        "🛠 **Problem Details** 🛠\n"
        f"   - **Title**: {problem['title']}\n"
        f"   - **Abstract**: {problem['abstract']}\n"
        f"   - **Number**: {problem['number']}\n"
        f"   - **Comment**: {problem['comment']}\n\n"
        
        "🔍 **Top 3 Similar Anomalies** 🔍\n"
        )

        for i, anomaly in enumerate(top3_anomalies, 1):
            result += (
                f"   {i}. **Anomaly**\n"
                f"      - **Title**: {anomaly['title']}\n"
                f"      - **Description**: {anomaly['description']}\n"
                f"      - **Similarity Score**: {anomaly['similarity_score']:.2f}\n\n"
            )

        result += (
            "💡 **Recommendation to solve the issue** 💡\n"
            f"{recommendation_text}\n"
        )

        self.reset_pipeline()

        return result

    
    class AnomalyRetrievalAndRecommendationPipeline:
        @staticmethod
        async def run_pipeline(user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
     
            pipeline = Pipeline()
            await pipeline.on_startup()
            pipe_result =  pipeline.pipe(user_message, model_id, messages, body)
            await pipeline.on_shutdown()

            return pipe_result
