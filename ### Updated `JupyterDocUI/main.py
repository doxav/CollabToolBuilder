from synthesize_manager import SynthesisManager

def main():
    manager = SynthesisManager()
    notebook_path = "notebooks/Untitled.ipynb"

    # Generate the table of contents
    manager.generate_table_of_contents()

if __name__ == "__main__":
    main()
