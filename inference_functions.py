from langchain_core.messages import SystemMessage, HumanMessage

from config import openai_api_key, MODELS_CONFIG_LIST
from langchain_openai import ChatOpenAI

def generate_annotations(output: str, prompt_critic : str):
    """
    Ask a LLM to generate annotations on a prompt given in parameters.

    Parameters:
        output (str): The prompt to annotate.
        prompt_critic (str): The prompt for annotations management.

    Returns:
        str: The annotated prompt.
    """
    system_prompt = """
You're an AI assistant. Your task is to generate annotations on the prompt given to you. The output should be exactly the same as the input but with some annotations
in it, no changes on the text itself. The annotations will have this format:
'\\<approve/fix/delete>{{the text to annotate}}{{the feedback for the text (what is wrong, what is right, etc.)}}' 
You should not change anything of the content of the given prompt, only add annotations. You have to add 3 annotations, and for each one of them don't place 
them randomly, but place them in a way that they are relevant to the text.
Try to give real feedbacks for the annotations, and not just random feedbacks. Finally, don't annotate the same text twice or the full text in one; place annotations
on phrases or keywords that are relevant.

INPUT:
    """

    default_llm = ChatOpenAI(model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False, temperature=0.)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=output)]

    # Call the OpenAI API using the new client
    response = default_llm.invoke(messages)

    messages = [SystemMessage(content=prompt_critic), HumanMessage(content=response.content)]

    response = default_llm.invoke(messages)

    # Extract the assistant's reply
    final_output = response.content

    return final_output
