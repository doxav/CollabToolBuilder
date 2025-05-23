from typing import List, Union, Generator, Iterator, Dict
import json
import os
import re
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
 
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
        openai.api_key = self.valves.OPENAI_API_KEY

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def reset_pipeline(self):
        self.article_data = {}
        self.conversation_state = "start"

    def generateTexteFromInput(prompt_template="", text="", temperature=0.5, request_timout=120):
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from config import openai_api_key

        if prompt_template == "":
            prompt_template += """
            
        Extract the following key elements from the research paper provided below:
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


    def ask_next_question(self):
        if self.conversation_state == "start":
            return "Welcome to the pipeline. Please provide the title of the article."
        elif self.conversation_state == "ask_title":
            return "Ok, let's start! Please provide the title of the article."
        elif self.conversation_state == "ask_authors":
            return "Great! Now please provide the authors of the article separated by commas."
        elif self.conversation_state == "ask_abstract":
            return "Please provide a brief abstract of the article."
        elif self.conversation_state == "ask_confirmation":
            return (f"Here is the summary of the article:\n\n"
                    f"By: {', '.join(self.article_data['authors'])}\n"
                    f"Title: {self.article_data['title']}\n"
                    f"Abstract: {self.article_data['abstract']}\n\n"
                    f"Is this information correct? (y/n)")
        else:
            return "Thanks for using our pipeline. Please enter a whitespace to start again."

    def process_user_response(self, user_input):
        user_input = user_input.strip()

        if self.conversation_state == "start":
            self.conversation_state = "ask_title"
            return self.ask_next_question()

        elif self.conversation_state == "ask_title":
            self.article_data['title'] = user_input
            self.conversation_state = "ask_authors"
            return self.ask_next_question()

        elif self.conversation_state == "ask_authors":
            self.article_data['authors'] = [author.strip() for author in user_input.split(",")]
            self.conversation_state = "ask_abstract"
            return self.ask_next_question()

        elif self.conversation_state == "ask_abstract":
            self.article_data['abstract'] = user_input
            self.conversation_state = "ask_confirmation"
            return self.ask_next_question()

        elif self.conversation_state == "ask_confirmation":
            if user_input.lower() in ["yes", "y"]:
                self.conversation_state = "finished"
                return "Thanks! Let's start processing the article data."
            else:
                self.conversation_state = "start"
                self.article_data = {}
                return "Oh no! Let's start again. Please provide the title of the new article."

        elif user_input == "reset":
            self.reset_pipeline()
            return "The conversation has been reset."

        else:
            self.article_data = {}
            self.conversation_state = "ask_title"
            return "Let's start again. Please provide the title of the new article."

    def handle_conversation(self, user_input):
        return self.process_user_response(user_input)

    def generate_initial_outline(self, title: str, abstract: str) -> List[str]:
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=self.valves.OPENAI_API_KEY)

        prompt_template = """Generate potential sections for a research survey based on the title and abstract provided. 
        The title is "{title}" and the abstract is "{abstract}". 
        Ensure to create at least five sections that are relevant and coherent, suitable for a research survey paper."""

        prompter = ChatPromptTemplate.from_template(prompt_template)
        message = prompter.format_messages(title=title, abstract=abstract)

        generated_text = llm(message)
        return [section.strip() for section in generated_text.content.splitlines() if section.strip()]

    def generate_resources(self, outline: List[str], context: str) -> Dict[str, str]:
        resources = {}
        for section in outline:
            resource_summary = self.generateTexteFromInput(
                prompt_template=f"Extract and summarize key literature related to the section titled '{section}'.",
                text=context
            )
            resources[section] = resource_summary
        return resources

    def pipe(self, user_message: str, messages: str, body: dict, model_id: str) -> Union[str, Generator, Iterator]:
        if self.conversation_state != "finished":
            return self.handle_conversation(user_message)

        outline = self.generate_initial_outline(self.article_data['title'], self.article_data['abstract'])
        resources = self.generate_resources(outline, self.article_data['abstract'])

        # Step 4: Return the generated outline and resources in a suitable format
        return {
            "outline": outline,
            "resources": resources
        }

class OutlineGenerationPipeline:
    @staticmethod
    async def run_pipeline(user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message, model_id, messages, body)
        await pipeline.on_shutdown()
        return pipe_result
