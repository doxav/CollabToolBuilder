# import nbformat
# from nbformat.v4 import new_markdown_cell, new_code_cell
# from nbformat import write
# import re

# class SynthesisManager:
#     def __init__(self):
#         self.sections = []
#         self.resources = []

#     def load_notebook(self, notebook_path):
#         with open(notebook_path, 'r') as f:
#             return nbformat.read(f, as_version=4)

#     def save_notebook(self, notebook, notebook_path):
#         with open(notebook_path, 'w') as f:
#             write(notebook, f)


#     def update_notebook_from_sections_and_resources(self, notebook_path):
#         notebook = self.load_notebook(notebook_path)

#         for cell in notebook['cells']:
#             if cell['cell_type'] == 'markdown':
#                 if cell['source'].startswith('# Section'):
#                     section_id = int(cell['source'].split()[1])
#                     section_title = [section['title'] for section in self.sections if section['id'] == section_id]
#                     if section_title:
#                         cell['source'] = f"# Section {section_id}\n\n{section_title[0]}"
#                 elif 'resource_id' in cell.metadata:
#                     resource_id = cell.metadata['resource_id']
#                     resource_content = [resource['content'] for resource in self.resources if resource['id'] == resource_id]
#                     if resource_content:
#                         cell['source'] = resource_content[0]

#         self.save_notebook(notebook, notebook_path)


#     def create_and_add_section_then_return_id(self, notebook_path, title):
#         section_id = len(self.sections) + 1
#         self.sections.append({"id": section_id, "title": title})

#         notebook = self.load_notebook(notebook_path)
#         section_content = f"# Section {section_id}\n\nThis is the content of Section {section_id}."
#         new_cell = new_markdown_cell(section_content)

#         notebook['cells'].append(new_cell)
#         self.save_notebook(notebook, notebook_path)

#         return section_id

#     def edit_section(self, notebook_path, section_id, new_title):
#         notebook = self.load_notebook(notebook_path)

#         for cell in notebook['cells']:
#             if cell['cell_type'] == 'markdown' and cell['source'].startswith(f'# Section {section_id}'):
#                 cell['source'] = f"# Section {section_id}\n\n{new_title}"
#                 break

#         self.save_notebook(notebook, notebook_path)

#     def remove_section(self, notebook_path, section_id):
#         notebook = self.load_notebook(notebook_path)
#         notebook['cells'] = [cell for cell in notebook['cells'] if not (cell['cell_type'] == 'markdown' and cell['source'].startswith(f'# Section {section_id}'))]
#         self.save_notebook(notebook, notebook_path)
#         self.sections = [section for section in self.sections if section["id"] != section_id]

#     def swap_sections(self, notebook_path, section_id_1, section_id_2):
#         notebook = self.load_notebook(notebook_path)

#         for cell in notebook['cells']:
#             if cell['cell_type'] == 'markdown' and (cell['source'].startswith(f'# Section {section_id_1}') or cell['source'].startswith(f'# Section {section_id_2}')):
#                 cell['source'] = cell['source'].replace(f'{section_id_1}', 'TEMP_ID').replace(f'{section_id_2}', f'{section_id_1}').replace('TEMP_ID', f'{section_id_2}')

#         self.save_notebook(notebook, notebook_path)

#     def add_resource(self, notebook_path, content):
#         resource_id = len(self.resources) + 1
#         self.resources.append({"id": resource_id, "content": content})

#         notebook = self.load_notebook(notebook_path)
#         new_cell = new_markdown_cell(content)

#         notebook['cells'].append(new_cell)
#         self.save_notebook(notebook, notebook_path)

#         return resource_id

#     def edit_resource(self, notebook_path, resource_id, new_content):
#         notebook = self.load_notebook(notebook_path)

#         for cell in notebook['cells']:
#             if cell['cell_type'] == 'markdown' and cell.metadata.get('resource_id') == resource_id:
#                 cell['source'] = new_content
#                 break

#         self.save_notebook(notebook, notebook_path)

#     def remove_resource(self, notebook_path, resource_id):
#         notebook = self.load_notebook(notebook_path)
#         notebook['cells'] = [cell for cell in notebook['cells'] if cell.metadata.get('resource_id') != resource_id]
#         self.save_notebook(notebook, notebook_path)
#         self.resources = [resource for resource in self.resources if resource["id"] != resource_id]

#     def swap_resources(self, notebook_path, resource_id_1, resource_id_2):
#         notebook = self.load_notebook(notebook_path)

#         for cell in notebook['cells']:
#             if cell['cell_type'] == 'markdown':
#                 if cell.metadata.get('resource_id') == resource_id_1:
#                     cell.metadata['resource_id'] = resource_id_2
#                 elif cell.metadata.get('resource_id') == resource_id_2:
#                     cell.metadata['resource_id'] = resource_id_1

#         self.save_notebook(notebook, notebook_path)

#     def detect_notebook_changes(self, notebook_path):
#         updated_sections = []
#         updated_resources = []

#         notebook = self.load_notebook(notebook_path)

#         for cell in notebook['cells']:
#             if cell['cell_type'] == 'markdown':
#                 match = re.match(r'# Section (\d+)\n\n(.*)', cell['source'])
#                 if match:
#                     try:
#                         section_id = int(match.group(1))
#                         section_title = match.group(2).strip()
#                         updated_sections.append({"id": section_id, "title": section_title})
#                     except (ValueError, IndexError):
#                         # If extraction fails, print debug information
#                         print("Error extracting section information:")
#                         print("Cell source:", cell['source'])
#                         continue
#                 elif 'resource_id' in cell.metadata:
#                     resource_id = cell.metadata['resource_id']
#                     updated_resources.append({"id": resource_id, "content": cell['source']})

#         # Print the changes
#         print("Changes detected in the notebook:")
#         print("Updated Sections:")
#         for section in updated_sections:
#             print(f"Section ID: {section['id']}, Title: {section['title']}")

#         print("Updated Resources:")
#         for resource in updated_resources:
#             print(f"Resource ID: {resource['id']}, Content: {resource['content']}")


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

    def sync_notebook_changes(self, notebook_path):
        notebook = self.load_notebook(notebook_path)
        self.sections = []
        self.resources = []

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                match = re.match(r'# Section (\d+)\n\n(.*)', cell['source'])
                if match:
                    try:
                        section_id = int(match.group(1))
                        section_title = match.group(2).strip()
                        self.sections.append({"id": section_id, "title": section_title})
                    except (ValueError, IndexError):
                        # If extraction fails, print debug information
                        print("Error extracting section information:")
                        print("Cell source:", cell['source'])
                        continue
                elif 'resource_id' in cell.metadata:
                    resource_id = cell.metadata['resource_id']
                    self.resources.append({"id": resource_id, "content": cell['source']})

        self.save_notebook(notebook, notebook_path)

    def update_notebook_from_sections_and_resources(self, notebook_path):
        notebook = self.load_notebook(notebook_path)

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                if cell['source'].startswith('# Section'):
                    section_id = int(cell['source'].split()[1])
                    section_title = [section['title'] for section in self.sections if section['id'] == section_id]
                    if section_title:
                        cell['source'] = f"# Section {section_id}\n\n{section_title[0]}"
                elif 'resource_id' in cell.metadata:
                    resource_id = cell.metadata['resource_id']
                    resource_content = [resource['content'] for resource in self.resources if resource['id'] == resource_id]
                    if resource_content:
                        cell['source'] = resource_content[0]

        self.save_notebook(notebook, notebook_path)

    def create_and_add_section_then_return_id(self, notebook_path, title):
        section_id = len(self.sections) + 1
        self.sections.append({"id": section_id, "title": title})

        notebook = self.load_notebook(notebook_path)
        section_content = f"# Section {section_id}\n\nThis is the content of Section {section_id}."
        new_cell = new_markdown_cell(section_content)

        notebook['cells'].append(new_cell)
        self.save_notebook(notebook, notebook_path)

        return section_id

    def edit_section(self, notebook_path, section_id, new_title):
        notebook = self.load_notebook(notebook_path)

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and cell['source'].startswith(f'# Section {section_id}'):
                cell['source'] = f"# Section {section_id}\n\n{new_title}"
                break

        self.save_notebook(notebook, notebook_path)

    def remove_section(self, notebook_path, section_id):
        notebook = self.load_notebook(notebook_path)
        notebook['cells'] = [cell for cell in notebook['cells'] if not (cell['cell_type'] == 'markdown' and cell['source'].startswith(f'# Section {section_id}'))]
        self.save_notebook(notebook, notebook_path)
        self.sections = [section for section in self.sections if section["id"] != section_id]

    def swap_sections(self, notebook_path, section_id_1, section_id_2):
        notebook = self.load_notebook(notebook_path)

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and (cell['source'].startswith(f'# Section {section_id_1}') or cell['source'].startswith(f'# Section {section_id_2}')):
                cell['source'] = cell['source'].replace(f'{section_id_1}', 'TEMP_ID').replace(f'{section_id_2}', f'{section_id_1}').replace('TEMP_ID', f'{section_id_2}')

        self.save_notebook(notebook, notebook_path)

    def add_resource(self, notebook_path, content):
        resource_id = len(self.resources) + 1
        self.resources.append({"id": resource_id, "content": content})

        notebook = self.load_notebook(notebook_path)
        new_cell = new_markdown_cell(content)

        notebook['cells'].append(new_cell)
        self.save_notebook(notebook, notebook_path)

        return resource_id

    def edit_resource(self, notebook_path, resource_id, new_content):
        notebook = self.load_notebook(notebook_path)

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and cell.metadata.get('resource_id') == resource_id:
                cell['source'] = new_content
                break

        self.save_notebook(notebook, notebook_path)

    def remove_resource(self, notebook_path, resource_id):
        notebook = self.load_notebook(notebook_path)
        notebook['cells'] = [cell for cell in notebook['cells'] if cell.metadata.get('resource_id') != resource_id]
        self.save_notebook(notebook, notebook_path)
        self.resources = [resource for resource in self.resources if resource["id"] != resource_id]

    def swap_resources(self, notebook_path, resource_id_1, resource_id_2):
        notebook = self.load_notebook(notebook_path)

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                if cell.metadata.get('resource_id') == resource_id_1:
                    cell.metadata['resource_id'] = resource_id_2
                elif cell.metadata.get('resource_id') == resource_id_2:
                    cell.metadata['resource_id'] = resource_id_1

        self.save_notebook(notebook, notebook_path)

    def detect_notebook_changes(self, notebook_path):
        updated_sections = []
        updated_resources = []

        notebook = self.load_notebook(notebook_path)

        for cell in notebook['cells']:
            print("Cell: ", cell)
            if cell['cell_type'] == 'markdown':
                match = re.match(r'# Section (\d+)\n\n(.*)', cell['source'])
                
                if match:
                    try:
                        section_id = int(match.group(1))
                        section_title = match.group(2).strip()
                        updated_sections.append({"id": section_id, "title": section_title})
                    except (ValueError, IndexError):
                        # If extraction fails, print debug information
                        print("Error extracting section information:")
                        print("Cell source:", cell['source'])
                        continue
                elif 'resource_id' in cell.metadata:
                    resource_id = cell.metadata['resource_id']
                    updated_resources.append({"id": resource_id, "content": cell['source']})

        # Print the changes
        print("Changes detected in the notebook:")
        print("Updated Sections:")
        for section in updated_sections:
            print(f"Section ID: {section['id']}, Title: {section['title']}")

        print("Updated Resources:")
        for resource in updated_resources:
            print(f"Resource ID: {resource['id']}, Content: {resource['content']}")

