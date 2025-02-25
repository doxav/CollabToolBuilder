import datetime
import socket

from langchain_openai import ChatOpenAI

from config import MODELS_CONFIG_LIST, embedding_function
from utils.llm_utils import HumanLLM, UnifiedVectorDB
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
        response = self.human_llm_monitor.CallHumanLLM(
            system_prompt_template="tests/marseille",
            user_message=user_message.content,
            return_message_content_only=True,
            stream_output=False,
            prompt_directory="../prompts"
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
        response = self.human_llm_monitor.CallHumanLLM(
            system_prompt_template="tests/paris",
            user_message=user_message.content,
            return_message_content_only=True,
            stream_output=False,
            prompt_directory="../prompts"
        )
        return response

# Instanciation et utilisation du nouvel agent
if __name__ == "__main__":

    unique_id = f"{socket.gethostname()}_{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    if unique_id is not False and UnifiedVectorDB.unique_collection_id is None:
        UnifiedVectorDB.set_unique_collection_id(unique_id)
    # Initialize HumanLLMMonitor databases
    HumanLLM._check_and_init_vector_db(embedding_function=embedding_function)
    HumanLLM.check_init_class_db(force=True)

    HumanLLM.use_websocket = True
    # Initialize WebSocket server if used
    if HumanLLM.use_websocket:
        if HumanLLM.websocket_server is None:
            HumanLLM.initialize_websocket_server()

    # Initialiser un agent personnalisé
    llm_list = {
        "default_llm": ChatOpenAI(
            model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False,
            temperature=0.),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"], cache=False, temperature=0.)}
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
