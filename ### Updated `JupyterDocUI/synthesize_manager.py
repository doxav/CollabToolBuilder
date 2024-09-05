import nbformat
from nbformat.v4 import new_markdown_cell
from nbformat import write
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import OPENAI_API_KEY

class SynthesisManager:
    def __init__(self):
        self.sections = []
        self.resources = []

    def load_notebook(self, notebook_path):
        with open(notebook_path, 'r') as f:
            return nbformat.read(f, as_version=4)

    def save_notebook(self, notebook, notebook_path):
        with open(notebook_path, 'w') as f:
            write(notebook, f)

    def generate_outline_for_survey_paper(self, title, abstract, temperature=0.7):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=OPENAI_API_KEY)
        prompt_template = """Generate a structured outline for a research survey paper titled "{title}" based on the following abstract: "{abstract}". 
        Include sections such as Introduction, Literature Review, Methodology, Results, Discussion, and Conclusion, along with relevant sub-sections."""
        
        prompter = ChatPromptTemplate.from_template(prompt_template)
        message = prompter.format_messages(title=title, abstract=abstract)
        
        generated_text = llm(message)

        # Split the generated text into sections
        sections = generated_text.split('\n')
        for section in sections:
            if section.strip():  # Avoid empty lines
                section_title = section.strip()
                self.create_and_add_section_then_return_id(title=section_title, content="")

        # Log the event for creating the outline
        self.add_event(event="outline_created", data={"title": title, "abstract": abstract})

        return self

    def create_and_add_section_then_return_id(self, title: str, content: str, section_id: int = None, parent_id: int = None) -> int:
        section_id = len(self.sections) + 1
        self.sections.append({"id": section_id, "title": title, "content": content})
        return section_id

    def add_event(self, event: str, data: dict):
        print(f"Event: {event}, Data: {data}")
