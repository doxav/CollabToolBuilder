import json

from langchain_openai import ChatOpenAI
from config import MODELS_CONFIG_LIST
from learn import create_Nmajority_chain
from utils.llm_utils import HumanLLMMonitor

embedding_function = "intfloat/e5-base-v2" 
reset_db_indices = False

llmORchains_list = {
    "default_llm": ChatOpenAI(
        model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False,
        temperature=0.),
    "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"], cache=False, temperature=0.),
    #"default_llm": ChatGroq(model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False),#, temperature=0.),
    #"premium_llm": ChatGroq(model_name=MODELS_CONFIG_LIST["code_gpt"], cache=False),#, temperature=0.),
    "3_majority_chain": create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                                                reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"], num_models=3),
    "10_majority_chain": create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                                                reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"], num_models=10)
}

HumanLLMMonitor._check_and_init_vector_db(embedding_function=embedding_function, reset_db_indices=reset_db_indices)
HumanLLMMonitor.check_init_class_db(force=True)
answer = HumanLLMMonitor(llmORchains_list=llmORchains_list, agent_name='Validation_Agent').get_learnt_tasks()
print(type(answer))
print(len(answer))
answer = list(answer)
for ans in answer:
    task: dict = json.loads(ans)
    print(task['main_function_name'])