# SynthesisManager.py

class SynthesisManager:
    def __init__(self):
        self.sections = []
        self.resources = []

    def create_and_add_section_then_return_id(self, title):
        section_id = len(self.sections) + 1
        self.sections.append({"id": section_id, "title": title})
        return section_id

    def edit_section(self, section_id, new_title):
        for section in self.sections:
            if section["id"] == section_id:
                section["title"] = new_title

    def remove_section(self, section_id):
        self.sections = [section for section in self.sections if section["id"] != section_id]

    def swap_sections(self, section_id_1, section_id_2):
        # Simplified swap logic for illustration
        for section in self.sections:
            if section["id"] == section_id_1:
                section["id"] = section_id_2
            elif section["id"] == section_id_2:
                section["id"] = section_id_1

    def add_resource(self, content):
        resource_id = len(self.resources) + 1
        self.resources.append({"id": resource_id, "content": content})
        return resource_id

    def edit_resource(self, resource_id, new_content):
        for resource in self.resources:
            if resource["id"] == resource_id:
                resource["content"] = new_content

    def remove_resource(self, resource_id):
        self.resources = [resource for resource in self.resources if resource["id"] != resource_id]

    def swap_resources(self, resource_id_1, resource_id_2):
        # Simplified swap logic for illustration
        for resource in self.resources:
            if resource["id"] == resource_id_1:
                resource["id"] = resource_id_2
            elif resource["id"] == resource_id_2:
                resource["id"] = resource_id_1
