import os
from aider import io, coders, models

class AiderLLM:
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialise l'instance avec le nom du modèle spécifié (par défaut 'gpt-4o-mini')."""
        self.code_directory = None
        self.model_name = model_name  # Model can now be set dynamically
        self.aider_instance = None

    def set_code_directory(self, directory):
        """Initialise le répertoire de code à utiliser par Aider et prépare l'instance de génération."""
        self.code_directory = directory
        io_instance = io.InputOutput(yes=True, input_history_file="/dev/null", chat_history_file="/dev/null")
        # Instantiate Coder directly with model name
        self.aider_instance = coders.Coder.create(main_model=models.Model(self.model_name), io=io_instance)
        self.aider_instance.max_reflections = 4
        #self.aider_instance.show_announcements()

    def invoke(self, prompt):
        """Méthode d'invocation pour générer ou éditer du code en fonction du prompt."""
        # If aider_instance is not set, default to current working directory or None
        if not self.aider_instance:
            self.set_code_directory("/dev/null" if not self.code_directory else self.code_directory)

        if not self.aider_instance:
            raise ValueError("Code directory not set for Aider")

        prompt = f"SYSTEM MESSAGE: {prompt[0].content}\n\nUSER MESSAGE: {prompt[1].content}"
        return self.aider_instance.run(prompt)

    def with_structured_output(self, output_schema=None):
        # Aider ne supporte pas nativement la sortie structurée, mais on peut gérer cela en post-traitement si nécessaire
        return self

    def with_config(self, configurable=None):
        # Not yet implemented
        print("Configuration not yet implemented for Aider")
        return self
