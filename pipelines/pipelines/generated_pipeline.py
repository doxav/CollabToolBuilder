from typing import List, Dict, Union, Callable, Any
import os
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str

    def __init__(self):
        self.reset_pipeline()
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            }
        )

    def reset_pipeline(self):
        self.documents = None
        self.index = None
        self.conversation_state = "start"

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def register_functions(self) -> Dict[str, Callable]:
        return {
            "generateTextFromInput": self.generateTextFromInput,
            "generate_outline": self.generate_outline,
        }

    # Define the extracted functions as methods
    def generateTextFromInput(prompt_template = "", text="", temperature=0.5, request_timout=120):
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from config import openai_api_key
    
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
    
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=openai_api_key)
        prompter = ChatPromptTemplate.from_template(prompt_template)
        message = prompter.format_messages(text=text)
        generated_text = llm(message)
        return generated_text.content
    def generate_outline(bot, title, abstract, temperature=0.7):
        from env.IR_CPS_TechSynthesis.env import SynthesisManager
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from config import OPENAI_API_KEY
        import re
    
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=OPENAI_API_KEY)
        prompt_template = """Generate LaTeX code for a 15-page research survey document with bibliography. 
        The title of the research survey is "{title}," and the abstract is "{abstract}." 
        Include sections such as Introduction, Literature Review, Methodology, Findings, Discussion, 
        Conclusion, and Future Work. Ensure proper formatting and structure, and leave placeholders 
        for content in each section.
    """
        prompter = ChatPromptTemplate.from_template(prompt_template)
        message = prompter.format_messages(title=title, abstract=abstract)
        
        generated_text = llm(message)
    
    
        # Define potential keywords for sections and map them to standard section titles
        section_keywords = {
            'introduction': ['background', 'introduction', 'overview', 'state-of-the-art', 'survey'],
            'methodology': ['methodology', 'methods', 'approach', 'framework', 'strategies', 'skills'],
            'discussion': ['discussion', 'analysis', 'results', 'findings', 'challenges', 'limitations'],
            'conclusion': ['conclusion', 'summary', 'implications', 'future work', 'trends'],
        }
    
        # Initialize a dictionary to hold the identified sections
        potential_sections = {
            'introduction': '',
            'methodology': '',
            'discussion': '',
            'conclusion': '',
        }
        
        # Split abstract into sentences to find keywords for sections
        sections = re.split(r'\\section{', generated_text.content)
        for sentence in sections:
            for section_title, keywords in section_keywords.items():
                if any(keyword in sentence.lower() for keyword in keywords):
                    potential_sections[section_title] += sentence
    
        # Trim whitespace and remove empty sections
        potential_sections = {k: v.strip() for k, v in potential_sections.items() if v}
    
        # Create the title section
        bot.create_and_add_section_then_return_id(title=title, content="")
    
        # Create standard sections with the identified content
        for section_title, content in potential_sections.items():
            bot.create_and_add_section_then_return_id(title=section_title.capitalize(), content=content)
    
        # Log the event for creating the outline
        bot.add_event(event="outline_created", data={"title": title, "sections": potential_sections})
    
        # Return the structured outline with section titles
        return {section: content for section, content in potential_sections.items() if content}
    def pipe(self, user_message: str, **kwargs) -> Union[str, None]:
        functions = self.register_functions()
        # Dynamically call the function based on user_message
        for func_name, func in functions.items():
            if func_name in user_message.lower():
                # Call the function with provided arguments
                try:
                    return func(**kwargs)
                except TypeError as e:
                    return f"Error calling function {func_name}: {str(e)}"
        return "No matching function found for the provided message."

class GenericPipeline:
    @staticmethod
    async def run_pipeline(user_message: str, **kwargs) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message, **kwargs)
        await pipeline.on_shutdown()
        return pipe_result
