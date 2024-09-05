from synthesize_manager import SynthesisManager

def main():
    # Instantiate SynthesisManager
    manager = SynthesisManager()

    # Example usage of the new function
    title = "Innovative Approaches in AI"
    abstract = "This paper explores the latest advancements in artificial intelligence, focusing on novel techniques and applications across various domains."
    
    manager.generate_outline_for_survey_paper(title, abstract)

if __name__ == "__main__":
    main()
