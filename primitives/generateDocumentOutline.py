from env.IR_CPS_TechSynthesis.env import SynthesisManager

def generate_outline(bot: SynthesisManager, title, abstract, temperature=0.7):
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from config import OPENAI_API_KEY
    import re

    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, openai_api_key=OPENAI_API_KEY)
    prompt_template = """Generate LaTeX code for a 15-page research survey document with bibliography. 
    The title of the research survey is "{title}," and the abstract is "{abstract}." 
    Include sections such as Introduction, Literature Review, Methodology, Findings, Discussion, 
    Conclusion, and Future Work. Ensure proper formatting and structure, and leave placeholders 
    for content in each section.
"""
    prompter = ChatPromptTemplate.from_template(prompt_template)
    message = prompter.format_messages(title=title, abstract=abstract)
    
    generated_text = llm(message)


    # Define potential keywords for sections and map them to standard section titles
    section_keywords = {
        'introduction': ['background', 'introduction', 'overview', 'state-of-the-art', 'survey'],
        'methodology': ['methodology', 'methods', 'approach', 'framework', 'strategies', 'skills'],
        'discussion': ['discussion', 'analysis', 'results', 'findings', 'challenges', 'limitations'],
        'conclusion': ['conclusion', 'summary', 'implications', 'future work', 'trends'],
    }

    # Initialize a dictionary to hold the identified sections
    potential_sections = {
        'introduction': '',
        'methodology': '',
        'discussion': '',
        'conclusion': '',
    }
    
    # Split abstract into sentences to find keywords for sections
    sections = re.split(r'\\section{', generated_text.content)
    for sentence in sections:
        for section_title, keywords in section_keywords.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                potential_sections[section_title] += sentence

    # Trim whitespace and remove empty sections
    potential_sections = {k: v.strip() for k, v in potential_sections.items() if v}

    # Create the title section
    bot.create_and_add_section_then_return_id(title=title, content="")

    # Create standard sections with the identified content
    for section_title, content in potential_sections.items():
        bot.create_and_add_section_then_return_id(title=section_title.capitalize(), content=content)

    # Log the event for creating the outline
    bot.add_event(event="outline_created", data={"title": title, "sections": potential_sections})

    # Return the structured outline with section titles
    return {section: content for section, content in potential_sections.items() if content}
