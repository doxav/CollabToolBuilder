import time
import os

class NotebookSyncManager:
    def __init__(self, synthesis_manager, notebook_path):
        self.synthesis_manager = synthesis_manager
        self.notebook_path = notebook_path
        self.last_modified_time = self.get_last_modified_time()

    def get_last_modified_time(self):
        return os.path.getmtime(self.notebook_path)

    def check_notebook_changes(self):
        current_modified_time = self.get_last_modified_time()
        if current_modified_time > self.last_modified_time:
            self.last_modified_time = current_modified_time
            self.update_synthesis_manager()

    def update_synthesis_manager(self):
        notebook = self.synthesis_manager.load_notebook(self.notebook_path)
        self.synthesis_manager.sections = self.extract_sections(notebook)
        self.synthesis_manager.resources = self.extract_resources(notebook)

    def extract_sections(self, notebook):
        sections = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and cell['source'].startswith("# Section"):
                section_id = int(cell['source'].split()[1])
                section_title = cell['source'].split("\n")[0].replace("# ", "")
                sections.append({"id": section_id, "title": section_title})
        return sections

    def extract_resources(self, notebook):
        resources = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and 'resource_id' in cell.metadata:
                resource_id = cell.metadata['resource_id']
                resources.append({"id": resource_id, "content": cell['source']})
        return resources


import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell
from nbformat import write

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

    def create_and_add_section_then_return_id(self, notebook_path, title):
        section_id = len(self.sections) + 1
        self.sections.append({"id": section_id, "title": title})

        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Create a new Markdown cell with the section content
        section_content = f"# Section {section_id}\n\nThis is the content of Section {section_id}."
        new_cell = new_markdown_cell(section_content)

        # Add the new cell to the notebook
        notebook['cells'].append(new_cell)

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)

        return section_id

    def edit_section(self, notebook_path, section_id, new_title):
        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Modify the title of the specified section
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and cell['source'].startswith(f'# Section {section_id}'):
                cell['source'] = f"# Section {section_id}\n\n{new_title}"
                break

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)

    def remove_section(self, notebook_path, section_id):
        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Remove the section from the notebook
        notebook['cells'] = [cell for cell in notebook['cells'] if not (cell['cell_type'] == 'markdown' and cell['source'].startswith(f'# Section {section_id}'))]

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)

        # Remove the section from the internal list
        self.sections = [section for section in self.sections if section["id"] != section_id]

    def swap_sections(self, notebook_path, section_id_1, section_id_2):
        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Swap the sections in the notebook
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and (cell['source'].startswith(f'# Section {section_id_1}') or cell['source'].startswith(f'# Section {section_id_2}')):
                cell['source'] = cell['source'].replace(f'{section_id_1}', 'TEMP_ID').replace(f'{section_id_2}', f'{section_id_1}').replace('TEMP_ID', f'{section_id_2}')

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)

    def add_resource(self, notebook_path, content):
        resource_id = len(self.resources) + 1
        self.resources.append({"id": resource_id, "content": content})

        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Create a new code cell with the resource content
        new_cell = new_markdown_cell(content)

        # Add the new cell to the notebook
        notebook['cells'].append(new_cell)

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)

        return resource_id

    def edit_resource(self, notebook_path, resource_id, new_content):
        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Find the resource cell and update its content
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and cell.metadata.get('resource_id') == resource_id:
                cell['source'] = new_content
                break

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)

    def remove_resource(self, notebook_path, resource_id):
        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Remove the resource cell
        notebook['cells'] = [cell for cell in notebook['cells'] if cell.metadata.get('resource_id') != resource_id]

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)

        # Remove the resource from the internal list
        self.resources = [resource for resource in self.resources if resource["id"] != resource_id]

    def swap_resources(self, notebook_path, resource_id_1, resource_id_2):
        # Load the notebook
        notebook = self.load_notebook(notebook_path)

        # Swap resource cells
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                if cell.metadata.get('resource_id') == resource_id_1:
                    cell.metadata['resource_id'] = resource_id_2
                elif cell.metadata.get('resource_id') == resource_id_2:
                    cell.metadata['resource_id'] = resource_id_1

        # Save the modified notebook
        self.save_notebook(notebook, notebook_path)
        
        
synthesis_manager = SynthesisManager()

# Path to your Jupyter notebook
notebook_path = "notebooks/Untitled.ipynb"

# Instantiate NotebookSyncManager
notebook_sync_manager = NotebookSyncManager(synthesis_manager, notebook_path)

# Main loop for synchronization
while True:
    print(notebook_sync_manager.check_notebook_changes())
    time.sleep(5)