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
