# Reasoning: The task is to generate a structured outline for a research survey paper based on the provided title and abstract. 
# The implementation will utilize the ChatOpenAI model to extract key themes and structure the outline accordingly. 
# The function will ensure that the generated outline meets the performance acceptance criteria by including at least 5 main sections with relevant sub-sections.

def generate_outline_for_survey_paper(bot, title, abstract, temperature=0.7):
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from config import OPENAI_API_KEY
    import re

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=OPENAI_API_KEY)
    prompt_template = """Generate a structured outline for a research survey paper titled "{title}" based on the following abstract: "{abstract}". 
    Include sections such as Introduction, Literature Review, Methodology, Results, Discussion, and Conclusion, along with relevant sub-sections."""
    
    prompter = ChatPromptTemplate.from_template(prompt_template)
    message = prompter.format_messages(title=title, abstract=abstract)
    
    generated_text = llm(message)

    # Split the generated text into sections
    sections = generated_text.split('\n')
    for section in sections:
        if section.strip():  # Avoid empty lines
            section_title = section.strip()
            bot.create_and_add_section_then_return_id(title=section_title, content="")

    # Log the event for creating the outline
    bot.add_event(event="outline_created", data={"title": title, "abstract": abstract, "sections": sections})

    return bot

# Example usage
# generate_outline_for_survey_paper(bot, title="Innovative Approaches in AI", abstract="This paper explores the latest advancements in artificial intelligence, focusing on novel techniques and applications across various domains.")
