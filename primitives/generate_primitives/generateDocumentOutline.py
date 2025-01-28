def generate_outline(bot, 
                     prompt_template="""
                    Generate a 15-page research survey in {generation_format}.
                    The title is "{title}" and the abstract is "{abstract}". 
                    Include sections for {section_titles} and follow this instruction for each section "{instruction_for_section_content}".""",
                     section_list=None,
                     instruction_for_section_content='Provide a detailed explanation of the topic',
                     generation_format='LaTeX output ensuring to use \section{} for separating each section of your answer',
                     section_regex=r'\\section\{([^}]*)\}(.*?)(?=\\section\{|$)'):
    from langchain.prompts import ChatPromptTemplate
    import re
    global llm

    # Extract title and abstract from the bot's document
    title, abstract = bot.document.title, bot.document.abstract

    # Default section list if not provided
    section_list = section_list or ['introduction', 'methodology', 'discussion', 'conclusion']    

    # Prepare section titles and format the prompt
    section_titles = ', '.join([section.capitalize() for section in section_list])
    prompter = ChatPromptTemplate.from_template(prompt_template)
    message = prompter.format_messages(title=title, abstract=abstract, section_titles=section_titles, instruction_for_section_content=instruction_for_section_content, generation_format=generation_format)
    
    # Generate LaTeX text using the language model
    generated_text = llm(message)
    print(f"TYPE OF generated_text:{type(generated_text)}")

    # Regular expression to capture section titles and content
    section_pattern = re.compile(section_regex, re.DOTALL)
    matches = section_pattern.findall(generated_text if type(generated_text) == str else generated_text.content)

    # Initialize dictionary to store section content
    extracted_sections = {section: '' for section in section_list}

    # Process each section found
    for raw_title, content in matches:
        normalized_title = raw_title.lower()
        if normalized_title in section_list:
            extracted_sections[normalized_title] += content.strip() + "\n\n"

    # Persist sections with content in bot's structure to allow proper task evaluation
    persisted_sections = {}
    for section_title, content in extracted_sections.items():
        if content.strip():
            section_id = bot.create_and_add_section_then_return_id(title=section_title.capitalize(), content=content.strip())
            persisted_sections[section_title.capitalize()] = section_id

    # Return the mapping of section titles to IDs
    return persisted_sections

# set a an optional main section
if __name__ == "__main__":
    from CollabFunctionsGPTCreator.config import OPENAI_API_KEY
    import os
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    llm_custom = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def llm(prompt):
        return llm_custom.invoke([SystemMessage(content=""), HumanMessage(content=prompt)] if isinstance(prompt, str) else prompt).content

    # Create a bot mockup object with functions used
    class Document:
        title = "A Survey of Machine Learning Techniques"
        abstract = "This paper presents a comprehensive survey of machine learning techniques, focusing on the most popular algorithms and their applications."
    class Bot:
        def __init__(self):
            self.document = Document()
            self.sections = {}
        def create_and_add_section_then_return_id(self, title, content):
            section_id = len(self.sections) + 1
            self.sections[section_id] = {'title': title, 'content': content}
            return section_id
    
    bot = Bot()

    generate_outline(bot=bot)

    print(f"Bot sections: {bot.sections}")