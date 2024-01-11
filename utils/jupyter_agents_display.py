import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript, FileLink
from ipywidgets import VBox, Button, Output, Text, Layout
from datetime import datetime
import uuid
import threading, os
from functools import partial

class AgentDisplayManager:
    tab = widgets.Tab()
    task_selector = widgets.Dropdown(options=[], value=None, description='Select Task:', disabled=False,)
    outputs = {}  # This will now reference Accordion widgets instead of Output widgets
    tab_update_indicators = []
    element_counters = {}  # Class variable to keep track of the number of elements for each agent

    @classmethod
    def export_to_html(cls, file_path='agent_display.html'):
        html_content = "<html><body>\n"

        # Iterate over each agent and their messages
        for agent_name, accordion in cls.outputs.items():
            html_content += f"<h2>{agent_name}</h2>\n<ul>\n"
            for i in range(len(accordion.children)):
                widget = accordion.children[i]
                title = accordion.get_title(i)
                with widget:
                    # Assuming the first output is the one we're interested in
                    message = widget.outputs[0]['text']
                    html_content += f"<li><b>{title}:</b> {message}</li>\n"
            html_content += "</ul>\n"

        html_content += "</body></html>"

        # Write the HTML content to a file
        with open(file_path, 'w') as f:
            f.write(html_content)

        # Return a link for downloading the file
        return FileLink(file_path)
    
    @classmethod
    def add_agent(cls, agent_name):
        if agent_name not in cls.outputs:
            # Create an Accordion widget for each agent to hold messages
            accordion = widgets.Accordion(children=[])
            cls.outputs[agent_name] = accordion
            new_index = len(cls.tab.children)
            cls.tab.children = list(cls.tab.children) + [accordion]
            cls.tab.set_title(new_index, agent_name)
            cls.tab_update_indicators.append(False)
            cls.element_counters[agent_name] = 0  # Initialize counter for the new agent

    @classmethod
    def write_to_agent(cls, agent_name, message, element_name=None, append=False, reverse=True):
        if agent_name not in cls.outputs:
            cls.add_agent(agent_name)
        accordion = cls.outputs[agent_name]

        if append and accordion.children:
            # If appending, get the last Output widget and append the message
            output = accordion.children[0 if reverse else -1]
        else:
            # Create a new Output widget for the message
            output = widgets.Output()
            if reverse:
                # Memorize the current titles because inserting a new element will reset them
                old_titles = [accordion.get_title(i) for i in range(len(accordion.children))]
                accordion.children = [output] + list(accordion.children)
                # Reapply the old titles to their new positions and set the new title
                for i, title in enumerate(old_titles, start=1):  # Start from 1 as 0 will be the new element
                    accordion.set_title(i, title)
            else:
                accordion.children = list(accordion.children) + [output]
            if element_name is None:
                element_name = "New message"
            accordion.set_title(0 if reverse else (len(accordion.children) - 1), f"{len(accordion.children)}. {element_name}") 

        with output:
            if append:
                print(message, end="", flush=True)
            else:
                print(message)

        accordion.selected_index = (0 if reverse else (len(accordion.children) - 1))
        cls._indicate_tab_update(agent_name)
        # Activate the tab corresponding to the agent
        cls.tab.selected_index = list(cls.outputs.keys()).index(agent_name)

    @classmethod
    def _indicate_tab_update(cls, agent_name):
        if agent_name in cls.outputs:
            # Loop through each tab to find the one with the matching title
            for index in range(len(cls.tab.children)):
                if cls.tab.get_title(index) == agent_name:
                    if cls.tab.selected_index != index:
                        cls.tab_update_indicators[index] = True
                        cls.update_tab_titles()
                    break  # Exit the loop once the matching tab is found


    @classmethod
    def update_tab_titles(cls):
        for index in range(len(cls.tab.children)):  # Iterate over the index of each tab
            title = cls.tab.get_title(index)  # Use the get_title method
            indicator = "*" if cls.tab_update_indicators[index] else ""
            cls.tab.set_title(index, f"{indicator}{title}{indicator}")

    @classmethod
    def on_tab_change(cls, change):
        cls.tab_update_indicators[change.new] = False
        cls.update_tab_titles()

    @classmethod
    def add_new_task(cls, task_name):
        if task_name not in cls.task_selector.options:
            cls.task_selector.options = cls.task_selector.options + (task_name,)
            cls.task_selector.style.button_color = 'lightgreen'

    @classmethod
    def display(cls):
        display(cls.task_selector)
        display(cls.tab)

    @classmethod
    def display_export_html_interface(cls):
        # Create input widgets for file name and path
        name_with_time = "experiment_" + str(datetime.now().strftime("%y%m%d_%H%M")) +".html"
        file_name_input = widgets.Text(
            value=name_with_time,
            description='File name:',
            continuous_update=False
        )
        file_path_input = widgets.Text(
            value='./',
            description='Path:',
            continuous_update=False
        )
        export_button = widgets.Button(description="Export to HTML")

        # Define the action for the export button
        def on_export_button_clicked(b):
            file_path = os.path.join(file_path_input.value, file_name_input.value)
            file_link = cls.export_to_html(file_path)
            display(HTML(f"Download the file <a href='{file_link}' target='_blank'>here</a>."))

        export_button.on_click(on_export_button_clicked)

        # Display the widgets
        display(widgets.HBox([file_name_input, file_path_input, export_button]))
        export_button.on_click(partial(on_export_button_clicked))

    @classmethod
    def get_input(cls, agent_name, prompt):
        if agent_name not in cls.outputs:
            cls.add_agent(agent_name)
        accordion = cls.outputs[agent_name]

        # Create widgets
        input_text = Text(description=prompt)
        submit_btn = Button(description="Submit")
        output = Output()  # To display the prompt and the entered value

        user_input = [None]  # To capture the user's input

        def on_submit(btn=None):
            with output:
                user_input[0] = input_text.value
                print(f"You entered: {user_input[0]}")

        submit_btn.on_click(on_submit)
        input_text.on_submit(on_submit)

        # Display widgets
        input_box = VBox([output, input_text, submit_btn], layout=Layout(flex_flow='column'))
        accordion.children = list(accordion.children) + [input_box]
        section_index = len(accordion.children) - 1
        accordion.set_title(section_index, "Input")

        # Set this accordion section to be selected and opened
        accordion.selected_index = section_index

        # Activate the tab corresponding to the agent
        cls.tab.selected_index = list(cls.outputs.keys()).index(agent_name)

        # Use this to display the prompt message
        with output:
            print(prompt)

        # This approach doesn't block; you might need to fetch user_input[0] later
        return user_input[0]

