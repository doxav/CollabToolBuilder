from utils.file_utils import load_from_pickle, save_to_pickle
@load_from_pickle
@save_to_pickle
# This function returns a text generated for a given task on a text by GPT3.5 Given a prompt template and a text in order to summarize it, or extract some key information...
def generateTextFromInput(prompt_template = "", text="", temperature=0.5, request_timout=120):
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from config import OPENAI_API_KEY

    if prompt_template == "":
        prompt_template = """Extract the following key elements from the research paper provided below:
1. Abstract: Summarize the abstract and identify any key elements that are missing which are later provided in the introduction.
2. Conclusion: Summarize the conclusion of the paper.
3. Findings: Detail the main findings of the paper.
4. Challenges/Discussion: Highlight the main challenges or discussion points mentioned in the paper.
5. Methodology: Describe the methodology used in the paper.

The output should be in JSON format with the following keys (if any of the below elements are not present in the paper, the value for the respective JSON key should be 'not found'):
- 'abstract_and_missing_elements': Max length of 500 words.
- 'conclusion': Max length of 300 words.
- 'findings': Max length of 500 words.
- 'challenges_discussion': Max length of 400 words.
- 'methodology': Max length of 400 words.

Research Paper Text: {text}"""

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, openai_api_key=openai_api_key)
    prompter = ChatPromptTemplate.from_template(prompt_template)
    message = prompter.format_messages(text=text)
    generated_text = llm(message)
    return generated_text.content
