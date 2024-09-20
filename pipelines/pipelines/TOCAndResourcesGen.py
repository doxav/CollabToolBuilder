from typing import List, Union, Generator, Iterator
import os
import openai
from pydantic import BaseModel
import re
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

    def reset_pipeline(self):
        self.article_data = {}
        self.conversation_state = "start"

    def ask_next_question(self):
        if self.conversation_state == "start":
            return "Welcome to the TOC and resource generation pipeline. Please provide the title of the article."
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
                return "Thanks! Let's start processing the article data to generate the TOC and resources."
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

    def generate_toc(self) -> str:
        toc_structure = {
            "Introduction": "This section introduces the topic and outlines the objectives of the survey.",
            "Literature Review": "This section reviews existing literature relevant to the survey topic.",
            "Methodology": "This section describes the methods used to gather information for the survey.",
            "Findings": "This section presents the key findings from the survey conducted.",
            "Discussion": "This section discusses the implications and significance of the findings.",
            "Conclusion": "This section summarizes the key points and suggests areas for future research."
        }

        formatted_toc = f"## Table of Contents for: {self.article_data['title']}\n\n"
        for section, description in toc_structure.items():
            formatted_toc += f"### {section}\n{description}\n\n"

        return formatted_toc.strip()

    def suggest_resources(self) -> List[str]:
        # Placeholder for resource suggestions
        return [
            "1. Author A. (2020). Title of Related Paper 1. Journal Name.",
            "2. Author B. (2021). Title of Related Paper 2. Journal Name.",
            "3. Author C. (2022). Title of Related Paper 3. Journal Name.",
            "4. Author D. (2023). Title of Related Paper 4. Journal Name.",
            "5. Author E. (2020). Title of Related Paper 5. Journal Name.",
            "6. Author F. (2021). Title of Related Paper 6. Journal Name.",
            "7. Author G. (2022). Title of Related Paper 7. Journal Name.",
            "8. Author H. (2023). Title of Related Paper 8. Journal Name.",
            "9. Author I. (2020). Title of Related Paper 9. Journal Name.",
            "10. Author J. (2021). Title of Related Paper 10. Journal Name."
        ]

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
            ) -> Union[str, Generator, Iterator]:
        if self.conversation_state != "finished":
            return self.handle_conversation(user_message)

        toc = self.generate_toc()
        resources = self.suggest_resources()

        output = toc + "\n## Suggested Resources:\n" + "\n".join(resources)

        assert output is not None, "Output can't be None"
        assert isinstance(output, str), "Output should be str"

        return output

class TOCAndResourcesGenerationPipeline:
    @staticmethod
    async def run_pipeline(user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        pipeline = Pipeline()
        await pipeline.on_startup()
        pipe_result = pipeline.pipe(user_message, model_id, messages, body)
        await pipeline.on_shutdown()
        return pipe_result