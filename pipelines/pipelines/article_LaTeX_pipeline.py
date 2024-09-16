from typing import List, Union, Generator, Iterator, Dict
import json
from pydantic import BaseModel
import os
import re
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
        global documents, index
        pass

    async def on_shutdown(self):
        # Actions to perform on server shutdown
        pass

    def reset_pipeline(self):
        self.documents = None
        self.index = None
        self.article_data = {}
        self.conversation_state = "start"

    def extract_title_and_abstract(self, user_input: str) -> Dict[str, str]:
        match = re.search(r"title: (.+?)\s+abstract: (.+)", user_input, re.IGNORECASE)
        if match:
            return {
                "title": match.group(1).strip(),
                "abstract": match.group(2).strip()
            }
        return {}

    def ask_next_question(self):
        if self.conversation_state == "start":
            return "Welcome on the pipeline. Please provide some data on the article you used to generate the outline."
        elif self.conversation_state == "ask_title":
            return "Ok let's start! Please provide the title of the article."
        elif self.conversation_state == "ask_abstract":
            return "Please provide a brief abstract of the article."
        elif self.conversation_state == "ask_confirmation":
            return f"Here is the summary of the article:\n\nTitle: {self.article_data['title']}\nAbstract: {self.article_data['abstract']}\n\nIs this information correct? (yes/no)"
        else:
            return "Thanks for using our pipeline. Please enter a whitespace to start again."

    def process_user_response(self, user_input):
        if user_input is None:
            self.reset_pipeline()
            return "Now can you help me with providing some information about the article you want to structure?"

        user_input = user_input.strip()

        if self.conversation_state == "start" and user_input != "reset":
            self.conversation_state = "ask_title"
            return self.ask_next_question()

        elif self.conversation_state == "ask_title" and user_input != "reset":
            self.article_data['title'] = user_input
            self.conversation_state = "ask_abstract"
            return self.ask_next_question()

        elif self.conversation_state == "ask_abstract" and user_input != "reset":
            self.article_data['abstract'] = user_input
            self.conversation_state = "ask_confirmation"
            return self.ask_next_question()

        elif self.conversation_state == "ask_confirmation" and user_input != "reset":
            if user_input.lower() in ["yes", "y"]:
                self.conversation_state = "finished"
                return "Thanks! Let's start processing the article data. Please press enter to proceed."
            else:
                self.conversation_state = "ask_title"
                self.article_data = {}
                return "Oh no! Let's start again. Please provide the title of the new article."

        elif user_input == "reset" or user_input is None:
            self.reset_pipeline()
            return "The conversation has been reset."

        else:
            self.article_data = {}
            self.conversation_state = "ask_title"
            return "Let's start again. Please provide the title of the new article."

    def handle_conversation(self, user_input):
        return self.process_user_response(user_input)

    def generate_initial_outline(self, title: str, abstract: str, temperature=0.7) -> Dict[str, str]:
        # Using the function provided earlier to generate a LaTeX outline
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=openai.api_key)
        
        prompt_template = """Generate LaTeX code for a 15-page research survey document with bibliography. 
        The title of the research survey is "{title}," and the abstract is "{abstract}." 
        Include sections such as Introduction, Literature Review, Methodology, Findings, Discussion, 
        Conclusion, and Future Work. Ensure proper formatting and structure, and leave placeholders 
        for content in each section."""
        
        prompter = ChatPromptTemplate.from_template(prompt_template)
        message = prompter.format_messages(title=title, abstract=abstract)
        
        generated_text = llm(message)
        
        # Process the generated LaTeX text as needed
        return generated_text.content

    def pipe(self, user_message: str, messages: str, body: dict, model_id: str) -> Union[str, Generator, Iterator]:
        if self.conversation_state != "finished":
            return self.handle_conversation(user_message)

        article = {
            'title': self.article_data['title'],
            'abstract': self.article_data['abstract']
        }

        assert article['title'] is not None, "Title can't be None"
        assert article['abstract'] is not None, "Abstract can't be None"

        # Step 2: Generate the LaTeX outline based on the title and abstract
        outline = self.generate_initial_outline(article['title'], article['abstract'])

        # Step 4: Return the generated outline in a suitable format
        return outline

class OutlineGenerationPipeline:
    @staticmethod
    async def run_pipeline(user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message, model_id, messages, body)
        await pipeline.on_shutdown()
        return pipe_result
