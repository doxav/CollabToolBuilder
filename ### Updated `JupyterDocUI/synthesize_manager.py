import nbformat
from nbformat.v4 import new_markdown_cell
from nbformat import write
import re

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

    def create_and_add_section_then_return_id(self, title: str, content: str) -> int:
        section_id = len(self.sections) + 1
        self.sections.append({"id": section_id, "title": title, "content": content})
        return section_id

    def get_all_sections(self):
        return self.sections

    def generate_table_of_contents(self):
        formatted_toc = []
        sections = self.get_all_sections()

        for section in sections:
            level = 1  # Assuming all sections are top-level for simplicity
            formatted_toc.append(self.format_toc_entry(section['title'], level))

        toc_output = "\n".join(formatted_toc)
        self.create_and_add_section_then_return_id(title="Table of Contents", content=toc_output)
        self.add_event(event="toc_generated", data={"toc": toc_output})

    def format_toc_entry(self, title: str, level: int) -> str:
        return f"{'    ' * (level - 1)}- {title}"

    def add_event(self, event: str, data: dict):
        # Placeholder for event logging
        print(f"Event: {event}, Data: {data}")

# Other methods remain unchanged...
