# button_script.py

# Import your SynthesisManager class
from syn import SynthesisManager

# Import IPython
from IPython.display import display, HTML

# Create an instance of SynthesisManager
manager = SynthesisManager()

# Function to add a section and display the result
print("hi")
def add_section():
    section_id = manager.create_and_add_section_then_return_id("Introduction")
    display(HTML(f"<p>Section added: {manager.sections}</p>"))
