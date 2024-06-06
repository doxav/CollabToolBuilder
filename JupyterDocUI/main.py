
# # from kk import SynthesisManager

# # # Create an instance of SynthesisManager
# # manager = SynthesisManager()

# # # Function to add a section to the specified notebook
# # def add_section(notebook_path):
# #     section_id = manager.create_and_add_section_then_return_id(notebook_path, "Introduction")
# #     print(f"Section {section_id} added to the notebook at: {notebook_path}")

# # if __name__ == "__main__":
# #     # Provide the path to the notebook file
# #     notebook_path = "Untitled.ipynb"
# #     add_section(notebook_path)


# # main.py

# # from synthesizeManager.synthesize_manager import SynthesisManager
# from syn import SynthesisManager


# def main():
#     # Instantiate SynthesisManager
#     manager = SynthesisManager()

#     # Path to the notebook file
#     notebook_path = "notebooks/Untitled.ipynb"


#     # Create an instance of SynthesisManager
#     manager = SynthesisManager()

#     # Synchronize sections and resources from the notebook
#     manager.sync_sections_and_resources(notebook_path)

#     # Print the current sections and resources
#     print("Sections:", manager.sections)
#     print("Resources:", manager.resources)

#     # Modify the sections/resources (for example)
#     manager.sections.append({"id": len(manager.sections) + 1, "title": "New Section"})
#     manager.resources.append({"id": len(manager.resources) + 1, "content": "New Resource Content"})

#     # Update the notebook based on the modified sections/resources
#     manager.update_notebook_from_sections_and_resources(notebook_path)

#     # Confirm changes by synchronizing again and printing
#     manager.sync_sections_and_resources(notebook_path)
#     print("Sections after update:", manager.sections)
#     print("Resources after update:", manager.resources)

    
    
    
    # Add a section to the notebook
    # section_id = manager.create_and_add_section_then_return_id(notebook_path, "Introduction", 4, content='Aka')
    # print(f"Section {section_id} added to the notebook at: {notebook_path}")

    # # # Edit a section in the notebook
    # # manager.edit_section(notebook_path, section_id, "Updated Introduction")
    # # print(f"Section {section_id} edited in the notebook at: {notebook_path}")

    # # Remove a section from the notebook
    # manager.remove_section(notebook_path, section_id)
    # print(f"Section {section_id} removed from the notebook at: {notebook_path}")

    # # Swap sections in the notebook
    # section_id_1 = manager.create_and_add_section_then_return_id(notebook_path, "Section 1")
    # section_id_2 = manager.create_and_add_section_then_return_id(notebook_path, "Section 2")
    # manager.swap_sections(notebook_path, section_id_1, section_id_2)
    # print(f"Sections {section_id_1} and {section_id_2} swapped in the notebook at: {notebook_path}")
    # resource_id = manager.add_resource(notebook_path, "# Resource Title\n\nThis is the content of the resource.")
    # print(f"Resource {resource_id} added to the notebook at: {notebook_path}")

    # # Edit a resource in the notebook
    # manager.edit_resource(notebook_path, resource_id, "# Updated Resource Title\n\nThis is the updated content of the resource.")
    # print(f"Resource {resource_id} edited in the notebook at: {notebook_path}")

    # # Remove a resource from the notebook
    # manager.remove_resource(notebook_path, resource_id)
    # print(f"Resource {resource_id} removed from the notebook at: {notebook_path}")

    # # # Swap resources in the notebook
    # resource_id_1 = manager.add_resource(notebook_path, "# Resource 1\n\nContent of Resource 3")
    # resource_id_2 = manager.add_resource(notebook_path, "# Resource 2\n\nContent of Resource 4")
    # manager.swap_resources(notebook_path, resource_id_2, resource_id_1)
    # print(f"Resources {resource_id_1} and {resource_id_2} swapped in the notebook at: {notebook_path}")

# if __name__ == "__main__":
#     main()


# import time
# import os.path
# from syn import SynthesisManager

# # if __name__ == "__main__":
# #     notebook_path = "notebooks/Untitled.ipynb"

# #     # Create an instance of SynthesisManager
# #     manager = SynthesisManager()

# #     # Initial synchronization
# #     manager.sync_sections_and_resources(notebook_path)

# #     while True:
# #         # Get the last modification time of the notebook file
# #         last_modified_time = os.path.getmtime(notebook_path)

# #         # Check if the notebook file has been modified
# #         if last_modified_time > manager.last_sync_time:
# #             print("Notebook file has been modified. Synchronizing...")
# #             # Synchronize sections and resources from the notebook
# #             manager.sync_sections_and_resources(notebook_path)

# #             # Print the current sections and resources
# #             print("Sections:", manager.sections)
# #             print("Resources:", manager.resources)

# #             # Update the last sync time
# #             manager.last_sync_time = last_modified_time

# #         # Wait for a certain interval before checking again (e.g., every 5 seconds)
# #         time.sleep(5)
# # Create a Jupyter notebook
# notebook_path = "notebooks/Untitled.ipynb"

# # Instantiate a SynthesisManager object
# manager = SynthesisManager()

# # Create and add a section
# section_id = manager.create_and_add_section_then_return_id(notebook_path, "Introduction")

# # Add a resource
# resource_id = manager.add_resource(notebook_path, "This is a resource")

# # Edit the section title directly in the notebook
# # (Simulate manual changes made in the Jupyter notebook UI)
# # Open the notebook in Jupyter, modify the section title manually, then save the notebook

# # Sync changes from the notebook to the SynthesisManager instance
# manager.sync_notebook_changes(notebook_path)

# # Verify that changes are reflected in the SynthesisManager instance
# print(manager.sections)
# print(manager.resources)

from synthesize_manager import SynthesisManager
import time

def main():
    # Create an instance of SynthesisManager
    manager = SynthesisManager()

    # Specify the path to your Jupyter notebook file
    notebook_path = 'notebooks/Untitled.ipynb'

    while True:
        # Sync changes from the notebook to SynthesisManager
        manager.sync_notebook_changes(notebook_path)

        # Detect and print changes in the notebook
        manager.detect_notebook_changes(notebook_path)

        # Wait for some time before checking again
        time.sleep(3)  # Adjust the time interval as needed

if __name__ == "__main__":
    main()


