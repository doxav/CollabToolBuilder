from typing import List, Union, Generator, Iterator
import os
import openai
from pydantic import BaseModel
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json

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
            return "Welcome to the TOC and resource generation pipeline."
        elif self.conversation_state == "ask_title":
            return "Great, let's start! Now please provide the title of the article."
        elif self.conversation_state == "ask_abstract":
            return "Can you provide a brief abstract of the article?"
        elif self.conversation_state == "ask_authors":
            return "Now please provide the authors of the article, separated by commas."
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
            return "Now can you help me by providing some information about the article you want to structure?"

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

    def generate_section_content(self, section_title, authors, abstract, temperature=0.7):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=self.valves.OPENAI_API_KEY)
        prompt_template = f"Generate detailed content for the section titled '{section_title}' based on the following abstract:\n\n{abstract} by those {authors}"
        prompter = ChatPromptTemplate.from_template(prompt_template)
        message = prompter.format_messages(abstract=abstract)
        
        generated_text = llm(message)
        return generated_text.content

    def generate_toc(self) -> str:
        sections = [
            "Introduction",
            "Title",
            "Authors",
            "Abstract",
            "Literature Review",
            "Methodology",
            "Findings",
            "Discussion",
            "Conclusion"
        ]
        
        formatted_toc = f"## Table of Contents for: {self.article_data['title']}\n\n"
        content = {}

        # Generate section content
        for section in sections:
            if section == "Title":
                section_content = self.article_data['title']
            elif section == "Authors":
                section_content = ", ".join(self.article_data['authors'])
            elif section == "Abstract":
                section_content = self.article_data['abstract']
            else:
                section_content = self.generate_section_content(section,self.article_data['authors'], self.article_data['abstract'])
            
            content[section] = section_content
            formatted_toc += f"### {section_content}\n\n"

        return formatted_toc.strip()

    def suggest_resources(self) -> List[str]:
        title_keywords = re.findall(r'\b\w+\b', self.article_data['title'])
        abstract_keywords = re.findall(r'\b\w+\b', self.article_data['abstract'])

        resources = [
            f"1. Related work based on '{self.article_data['title']}' and its methodology.",
            f"2. Previous studies that explored topics similar to '{self.article_data['title']}'.",
            f"3. Research on areas discussed in the abstract: {' '.join(abstract_keywords[:3])}.",
            f"4. Review articles focusing on methodologies used in '{self.article_data['title']}'.",
            f"5. Studies by {self.article_data['authors'][0]} and other relevant authors."
        ]

        return resources

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        if self.conversation_state != "finished":
            return self.handle_conversation(user_message)

        toc = self.generate_toc()
        resources = self.suggest_resources()
        
        # Organize output
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
