import pytest
import datetime
import socket

from langchain_openai import ChatOpenAI

from config import MODELS_CONFIG_LIST, embedding_function
from utils.human_llm import HumanLLM, HumanLLMConfig
from utils.llm_utils import UnifiedVectorDB
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

# Déclaration d'un nouvel agent
class MarseillaisAgent:
    def __init__(self, agent_name, llmORchains_list=None):

        self.agent_name = agent_name
        if llmORchains_list is None:
            llmORchains_list = ["gpt-3.5-turbo"]  # Exemple de modèle par défaut
        self.human_llm_monitor = HumanLLM(
            agent_name=self.agent_name,
            llmORchains_list=llmORchains_list
        )
        self.human_llm_monitor.agent_name = self.agent_name  # Fix explicite


    def make_inference(self, user_message_content):
        # Construction des messages pour l'inférence
        user_message = HumanMessage(content=user_message_content)

        # Appel de l'inférence via HumanLLMMonitor
        response = self.human_llm_monitor.invoke(
            system_prompt_template="tests/marseille",
            user_message=user_message.content,
            return_message_content_only=True,
            stream_output=False,
            #prompt_directory="../prompts"
        )
        return response

# Déclaration d'un nouvel agent
class ParisienAgent:
    def __init__(self, agent_name, llmORchains_list=None):

        self.agent_name = agent_name
        if llmORchains_list is None:
            llmORchains_list = ["gpt-3.5-turbo"]  # Exemple de modèle par défaut
        self.human_llm_monitor = HumanLLM(
            agent_name=self.agent_name,
            llmORchains_list=llmORchains_list
        )
        self.human_llm_monitor.agent_name = self.agent_name  # Fix explicite


    def make_inference(self, user_message_content):
        # Construction des messages pour l'inférence
        user_message = HumanMessage(content=user_message_content)

        # Appel de l'inférence via HumanLLMMonitor
        response = self.human_llm_monitor.invoke(
            system_prompt_template="tests/paris",
            user_message=user_message.content,
            return_message_content_only=True,
            stream_output=False,
            #prompt_directory="../prompts"
        )
        return response

def test_marseillais_and_parisien_agents_inference():
    # # Initialize HumanLLMMonitor databases
    config = HumanLLMConfig()
    config.common_vectordb_config.embedding_function = embedding_function
    config.initialize()

    # Initialiser un agent personnalisé
    llm_list = {
        "default_llm": ChatOpenAI( model_name=MODELS_CONFIG_LIST.getattr("basic_gpt", "gpt-4.1-nano") if MODELS_CONFIG_LIST else "gpt-4.1-nano", cache=False, temperature=0.),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST.getattr("smart_gpt", "gpt-4.1-nano") if MODELS_CONFIG_LIST else "gpt-4.1-nano", cache=False, temperature=0.)}
    marseillais_agent = MarseillaisAgent("Marseillais", llmORchains_list=llm_list)
    parisien_agent = ParisienAgent("Parisien", llmORchains_list=llm_list)

    # Première inférence
    message1 = "Quelle est la capitale de la France ?"
    response1 = marseillais_agent.make_inference(message1)
    print(f"Réponse à la question '{message1}': {response1}")

    # Deuxième inférence
    message2 = "Qui est l'actuel président des États-Unis ?"
    response2 = parisien_agent.make_inference(message2)
    print(f"Réponse à la question '{message2}': {response2}")

# Instanciation et utilisation du nouvel agent
if __name__ == "__main__":
    test_marseillais_and_parisien_agents_inference()
    print("Inference test with fake agents Marseillais et Parisien terminés avec succès.")
