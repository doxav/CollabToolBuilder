from typing import List, Dict, Union
import os
import re
import openai
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
        openai.api_key = self.valves.OPENAI_API_KEY

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def reset_pipeline(self):
        self.article_data = {}
        self.conversation_state = "start"

    def ask_next_question(self):
        if self.conversation_state == "start":
            return "Welcome to the outline generation pipeline. Please provide the title of the article."
        elif self.conversation_state == "ask_title":
            return "Great! Now please provide the abstract of the article."
        elif self.conversation_state == "ask_authors":
            return "Now please provide the authors of the article separated by commas."
        elif self.conversation_state == "ask_confirmation":
            return (f"Here is the summary of the article:\n"
                    f"Title: {self.article_data['title']}\n"
                    f"Abstract: {self.article_data['abstract']}\n"
                    f"Authors: {', '.join(self.article_data['authors'])}\n"
                    f"Is this information correct? (y/n)")
        else:
            return "Thanks for using our pipeline. Please enter a whitespace to start again."

    def process_user_response(self, user_input):
        if user_input is None:
            self.reset_pipeline()
            return "Now can you help me with providing some information about the article you want to structure?"

        user_input = user_input.strip()

        if self.conversation_state == "start":
            self.conversation_state = "ask_title"
            return self.ask_next_question()

        elif self.conversation_state == "ask_title":
            self.article_data['title'] = user_input
            self.conversation_state = "ask_abstract"
            return self.ask_next_question()

        elif self.conversation_state == "ask_abstract":
            self.article_data['abstract'] = user_input
            self.conversation_state = "ask_authors"
            return self.ask_next_question()

        elif self.conversation_state == "ask_authors":
            self.article_data['authors'] = user_input.split(",")
            self.conversation_state = "ask_confirmation"
            return self.ask_next_question()

        elif self.conversation_state == "ask_confirmation":
            if user_input.lower() in ["yes", "y"]:
                self.conversation_state = "finished"
                return "Thanks! Let's start processing the article data to generate the outline."
            else:
                self.conversation_state = "ask_title"
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

    def generate_initial_outline(self, title: str, abstract: str, authors: List[str]) -> str:
        outline = {
            "Introduction": "This section introduces the topic and outlines the objectives of the survey.",
            "Literature Review": "This section reviews existing literature relevant to the survey topic.",
            "Methodology": "This section describes the methods used to gather information for the survey.",
            "Findings": "This section presents the key findings from the survey conducted.",
            "Discussion": "This section discusses the implications and significance of the findings.",
            "Conclusion": "This section summarizes the key points and suggests areas for future research."
        }

        formatted_outline = f"## Outline for the Research Survey Paper: {title}\n\n"
        for section, description in outline.items():
            formatted_outline += f"### {section}\n{description}\n\n"

        return formatted_outline.strip()

    def pipe(self, user_message: str) -> str:
        if self.conversation_state != "finished":
            return self.handle_conversation(user_message)

        outline = self.generate_initial_outline(
            self.article_data['title'],
            self.article_data['abstract'],
            self.article_data['authors']
        )

        assert outline is not None, "Outline can't be None"
        assert isinstance(outline, str), "Outline should be str"

        return outline

class OutlineGenerationPipeline:
    @staticmethod
    async def run_pipeline(user_message: str) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message)
        await pipeline.on_shutdown()
        return pipe_result