import os

from langchain_core.messages import SystemMessage, HumanMessage
from utils.human_llm import  HumanLLM

from config import MODELS_CONFIG_LIST
from langchain_openai import ChatOpenAI

def generate_annotations(output: str, output_id : int = None):
    """
    Ask a LLM to generate annotations on a prompt given in parameters.

    Parameters:
        output (str): The prompt to annotate.
        output_id (int): The id of the output to annotate.
    Returns:
        str: The annotated prompt.
    """

    base_dir = f"IR_CPS_TechSynthesis/{os.environ['NAME_XP']}"

    default_llm = ChatOpenAI(model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False, temperature=0.)

    messages = [SystemMessage(content=HumanLLM.load_prompt(prompt=f"{base_dir}/answer_annotate")), HumanMessage(content=output)]

    # Call the OpenAI API using the new client
    response = default_llm.invoke(messages)

    messages = [SystemMessage(content=HumanLLM.load_prompt(prompt=f"{base_dir}/apply_annotations")), HumanMessage(content=response.content)]

    response = default_llm.invoke(messages)

    # Extract the assistant's reply
    final_output = response.content

    return final_output
