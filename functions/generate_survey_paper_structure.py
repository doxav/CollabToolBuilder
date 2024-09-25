
def generateTextFromInput(prompt_template="", text="", temperature=0.5, request_timout=120):
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from config import openai_api_key

    if prompt_template == "":
        prompt_template = """Extract the following key elements from the research paper provided below:
1. Abstract: Summarize the abstract and identify any key elements that are missing which are later provided in the introduction.
2. Conclusion: Summarize the conclusion of the paper.
3. Findings: Detail the main findings of the paper.
4. Challenges/Discussion: Highlight the main challenges or discussion points mentioned in the paper.
5. Methodology: Describe the methodology used in the paper.

The output should be in JSON format with the following keys (if any of the below elements are not present in the paper, the value for the respective JSON key should be 'not found'):
- 'abstract_and_missing_elements': Max length of 500 words.
- 'conclusion': Max length of 300 words.
- 'findings': Max length of 500 words.
- 'challenges_discussion': Max length of 400 words.
- 'methodology': Max length of 400 words.

Research Paper Text: {text}"""

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=openai_api_key)
    prompter = ChatPromptTemplate.from_template(prompt_template)
    message = prompter.format_messages(text=text)
    generated_text = llm(message)
    return generated_text.content

def generate_outline(bot, title, abstract, temperature=0.7):
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from config import openai_api_key
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=openai_api_key)
    prompt_template = """Generate potential sections for a research survey based on the title and abstract provided. 
    The title is "{title}" and the abstract is "{abstract}". 
    Ensure to create at least five sections that are relevant and coherent, suitable for a research survey paper.
"""
    prompter = ChatPromptTemplate.from_template(prompt_template)
    message = prompter.format_messages(title=title, abstract=abstract)
    
    generated_text = llm(message)
    return generated_text.content.splitlines()

def generate_survey_paper_structure(bot):
    """
The function generates a structured survey paper outline based on the provided document context, ensuring that a coherent set of sections and relevant resources are curated for the research paper. It first derives the title and abstract from the document, then creates an outline with at least five sections. For each section, it summarizes key literature related to the topic to enrich the content. Finally, it adds these sections and their corresponding resources to the document and logs the generation event.
:param bot: An object that contains the document and methods for adding sections and logging events
:return: The updated bot object with the newly created sections and resources
"""
    import json

    # Step 1: Generate potential sections for the table of contents based on the document context
    title = bot.document.title
    context = bot.document.context  # This is the abstract
    
    # Generate the outline using the reusable code
    outline = generate_outline(bot, title, context)
    
    # Remove any empty sections and ensure at least 5 sections are generated
    outline = [section.strip() for section in outline if section.strip()]
    if len(outline) < 5:
        raise ValueError("Insufficient sections generated for the survey paper.")

    # Step 2: Curate relevant resources for the survey topic
    resources = []
    for section in outline:
        resource_summary = generateTextFromInput(
            prompt_template=f"Extract and summarize key literature related to the section titled '{section}'.",
            text=context
        )
        resources.append({section: resource_summary})

    # Step 3: Organize sections and resources into a coherent structure
    # Adding sections to the document in structured format
    for section_title in outline:
        bot.create_and_add_section_then_return_id(title=section_title, content="")

    # Adding curated resources as a new section
    bot.create_and_add_section_then_return_id(title="References", content=json.dumps(resources, indent=2))

    # Log the event for the generated structure
    bot.add_event(event="survey_structure_generated", data={"title": title, "outline": outline, "resources": resources})

    return bot