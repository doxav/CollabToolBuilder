import os
import time
import learn
from langchain_openai import ChatOpenAI
import optuna as opt
from config import MODELS_CONFIG_LIST

def objective(trial):

    modelVariation = trial.suggest_int("model_variation", 0, 1)

    # Define parameters for the coach

    role_priming = trial.suggest_categorical("role_priming", [
        "research assistant",
        "AI coach",
        "task optimizer",
        "technical synthesis expert",
        "knowledge engineer",
        "content curator", 
        "data analyst assistant"
    ])
    goal_definition= trial.suggest_categorical("goal_description", [
        "produce high quality technical synthesis",
        "analyse complex data sets",
        "analyse a a Neo4J graph",
        "solve JIRA issue, given a graph of a complete dataset of JIRA anomalies from Neo4J"
    ])

    goal_function = trial.suggest_categorical("goal_function", [
        "Assist users in their situation analysis and decision making with situation assesment",
        "Show same issues depending on semantic similarity with the current issue from the graph",
        "Propose different resolutions depending on the graph",
        "Generate structred phases to solve this issue",
        "Propose actions to avoid this issue in the future"
    ])

    task_format = trial.suggest_categorical("task_format", [
        "[verb] [quantity if applicable] [object] [tools] [detailed instructions and parameters]",
        "[action] [target] using [method] with [specifications]",
        "[operation] on [subject] utilizing [resources] following [guidelines]",
        "[do] [what] [how] [with what] [detailed instructions]"
    ])
    reasoning_steps = trial.suggest_int("reasoning_steps", 3, 10)
    specification_depth = trial.suggest_int("specification_depth", 2, 5)
    use_examples = trial.suggest_categorical("use_examples", [True, False])
    include_failed_tasks = trial.suggest_categorical("include_failed_tasks", [True, False])
    plan_depth = trial.suggest_int("plan_depth", 2, 4)
    available_commands_detail = trial.suggest_categorical("available_commands_detail", [
        "minimal",
        "moderate",
        "comprehensive",
        "detailed"
    ])
    task_complexity = trial.suggest_categorical("task_complexity", [
        "Task shouldn’t be too difficult to convert into Python code given available commands and learnt tasks.",
        "Task should balance complexity and feasibility for effective implementation.",
        "Task should challenge the LLM while remaining solvable with available resources.",
        "Task should involve multiple steps that require coordination among different functions.",
        "Task should leverage advanced features of the LLM to achieve superior results.",
        "Task should be modular, allowing parts of the solution to be reused in other contexts.",
        "Task should be scalable, capable of being applied to larger datasets or more complex scenarios."
    ])
    criteria_to_remove = trial.suggest_int("criteria_to_remove", 1, 7)
    
    auto_add_test_to_prompt_answer = trial.suggest_categorical("auto_add_test_to_prompt_answer", [False])
    test_criteria = """7) After proposing the task, you should provide a test of the function corresponding to this task
    a) Write a one liner python call to the main function for each "Document to be tested", this call should be designed to maximize the expected results for the "Document to be tested"
    b) Precede each one liner call with a line of comment in this form "# document #uuid usage test" (e.g. "#document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test"...) to indicate to which document the code of the next line applies to given its unique id
    c) Call to the main function uses bot as first required parameter, then provide parameters sepecific to the document for this function
    d) No more than one test for each document, the total number of function calls in this test list should be equal to the number of "Document to be tested"
    """ if not auto_add_test_to_prompt_answer else 'do not add test cases automatically.'

    structure_FA= trial.suggest_categorical( """
                        <elementId>	4:39c83641-ab1b-4426-95d3-6646f85ee75c:8
                        <id>	8
                        fan_animpact_rnt	[]
                        fan_batiment	Archiva
                        fan_categorie	Bug
                        fan_comments	['Fixed. Patch for this attached.', 'Applied.']
                        fan_description_anomalie	When using Internet Explorer 7, the "Managed Repositories" and "Proxied Repositories" buttons under Administration are not displayed.
                        fan_etat	Closed
                        fan_fa_origine	[]
                        fan_gravite_decision	Major
                        fan_intitule	Managed Repositories and Proxied Repositories buttons under Administration are not displayed when using Internet Explorer 7.
                        fan_numero_fa	12788643
                        fan_programme	Web Interface
                        fan_programme_pere	Apache
                        fan_responsable_declaration	dangelito
                        fan_responsable_realisation	evenisse
                        fan_stade_generateur_anomalie	Maintenance
                        fan_subtasks	[]
                """, [True, False])
    

    cypher_request = trial.suggest_categorical(""" CREATE INDEX FA_fan_desc IF NOT EXISTS FOR (fa:FicheAnomalie) ON (fa.fan_description_anomalie);
                CREATE INDEX Personne_id IF NOT EXISTS FOR (p:Personne) ON (p.id);
                CREATE INDEX Programme_programme IF NOT EXISTS FOR (p:Programme) ON (p.programme);

        // Créer des programmes
                MATCH (fa:FicheAnomalie) WHERE fa.fan_programme IS NOT NULL
                MERGE (prg:Programme {'programme: toString(fa.fan_programme)'});

                MATCH (fa:FicheAnomalie) WHERE fa.fan_programme_pere IS NOT NULL
                MERGE (prg:Programme {'programme: toString(fa.fan_programme_pere)'});

        // Associer les programmes aux fiches d'anomalie
                MATCH (fa:FicheAnomalie), (prg:Programme {'programme: fa.fan_programme'})
                MERGE (prg)-[r:A_FA]->(fa);

                MATCH (fa:FicheAnomalie), (prg:Programme {'programme: fa.fan_programme_pere'})
                MERGE (prg)-[r:A_FA]->(fa);

        // Créer des relations entre les programmes
                MATCH (fa:FicheAnomalie), (prg:Programme {'programme: fa.fan_programme'}), (prg_pere:Programme {'programme: fa.fan_programme_pere'})
                MERGE (prg_pere)-[r:A_PROGRAMME]->(prg);


        // Créer des personnes responsables
                CALL apoc.periodic.iterate(
                'MATCH (n:FicheAnomalie) WHERE n.fan_responsable_declaration IS NOT NULL RETURN n',
                'MERGE (m:Personne {'id: toString(n.fan_responsable_declaration)'})',
                {'batchSize: 100, parallel: true'}
                );

                CALL apoc.periodic.iterate(
                'MATCH (n:FicheAnomalie) WHERE n.fan_responsable_realisation IS NOT NULL RETURN n',
                'MERGE (m:Personne {'id: toString(n.fan_responsable_realisation)'})',
                {'batchSize: 100, parallel: true'}
                );

        // Associer les personnes aux fiches d'anomalie
                CALL apoc.periodic.iterate(
                'MATCH (n:FicheAnomalie) WHERE n.fan_responsable_declaration IS NOT NULL MATCH (m:Personne {'id: n.fan_responsable_declaration'}) RETURN n, m',
                'MERGE (m)-[r:DECLARE]->(n)',
                {'batchSize: 100, parallel: false'}
                );

                CALL apoc.periodic.iterate(
                'MATCH (n:FicheAnomalie) WHERE n.fan_responsable_realisation IS NOT NULL MATCH (m:Personne {'id: n.fan_responsable_realisation'}) RETURN n, m',
                'MERGE (m)-[r:REALISE]->(n)',
                {'batchSize: 100, parallel: false'}
                );
            """, [True, False])

    test_example = """5. Tests:
        ```python
        #document #72dc469b-63f8-4751-aab5-6db3d16fca3c usage test:
        your_main_function_name(args specific to document #72dc469b-63f8-4751-aab5-6db3d16fca3c...)

        # document #c0533337-1c5e-4091-ba7d-061ae409cda4 usage test:
        your_main_function_name(args specific to document #c0533337-1c5e-4091-ba7d-061ae409cda4...)``` """ if not auto_add_test_to_prompt_answer else ''


    # Construct the prompt based on the suggested parameters
    criteria_coach = [
        f"1) Reason in {reasoning_steps} steps to find out the best task to minimize distance to goal.",
        f"2) Task should be written in the form of '{task_format}'",
        f"3) Task will be converted into Python code given available commands, learnt tasks, use of LLM if required.",
        f"4) Task should be novel compared to learnt {f'and failed ' if include_failed_tasks else ''}tasks.",
        f"5) Develop key minimal elements of specification (acceptance criteria, best strategies to compare, performance tips to beat a LLM) to successfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Organize the requirements with clear indexing to a depth of {specification_depth}.",
        f"6) Tasks provided should be generic, not specific to given examples{', so the reasoning can mention examples but proposed task and plan should not mention any information related to examples' if use_examples else ''}.",
        f"7) After proposing the task, you should provide a test case of the function corresponding to this task for each example:",
            f"a) Write a one liner python call to the main function for each 'Document to be tested', this call should be designed to maximize the expected results for the 'Document to be tested'",
            f"b) Precede each one liner call with a line of comment in this form '# document #uuid usage test' (e.g. '#document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test'...) to indicate to which document the code of the next line applies to given its unique id",
            f"c) Call to the main function uses 'problem' as first required parameter, then provide parameters sepecific to the document for this function (do not provide document #uuid as parameter but title and context instead)",
            f"d) Generate only one test for each document, so the total number of function calls in this test list should be equal to the number of 'Document to be tested'",
            f"e) Your only source of data will be the Neo4j graph provide in the primitives. Do not use the date."
        ]

    del criteria_coach[criteria_to_remove - 1]

    for i in range(len(criteria_coach)):
        coach_text = "\n".join(criteria_coach[i])
    
    prompt_coach = [f"""
            ROLE: 
                You are a {role_priming} that defines tasks to {goal_definition} . 

            CONTEXT:
                JIRA anomalies are records of issues encountered in software development and project management processes. 
                These anomalies can include bugs, performance errors, connectivity problems, and other technical malfunctions. 
                Each anomaly is documented with specific details to facilitate its identification, analysis, and resolution.
                The objective of analyzing JIRA anomalies is to identify clusters of recurring problems, determine root causes,
                and recommend corrective actions to improve software quality, reduce costs, and minimize delays. 
                All of those anomalies are stored on a Neo4J's graph. Connection informations to this database will be communicated 
                to you via the primitives.{task_complexity}

                We aim to have a chat agent to help all the Jira's users at every stage to automate possible tasks and improve their productivity by solving easily issues. 
                Among other things, it should be able to :
                    {goal_function}
                

            TASK:
                Based on the current context, we need to code functions to help users to solve JIRA's anomalies by leveraging the graph database from Neo4J. This database will be your only source of information.
                First: propose 3 differents plans to best support a crisis response / intervention from end-to-end starting when a new Jira anomalie is detected up to end of correction operation and capitalization.
                Second: reason and identify the next best task to be converted to code by a LLM and made available as a function to a chat, it should provide the best value to target while not being too complex to 
                be converted to code in one pass using a LLM.
                You should propose to choose between 2 different next best task. Task could be either the most frequent and globaly impactful task to be automated throughout plans, or the highest priority and impactful task at the current status of the JIRA's issue correction (if this status is not provided, consider we are the beginning of the operation).
                Task shouldn’t be too difficult to be converted into a single Python function for automation given available commands and learnt tasks to automate.
                Consider and re-use the already developped learnt tasks, do not propose the same task except if improving it can be done and is better than any other task.
                Detail this task into a specification checklist.
                Considering this checklist, end by listing any mandatory information for code implementation unkown by an LLM (e.g. GPT-4) that should be provided by a human or a search engine.

            INPUT:
                I will provide you if there are:
                - Learnt tasks available (with information gain between 0 (minimum) and 1 (maximum) on plan's titles, and contents): ...
                 {f'- Failed tasks to learn that are too hard: ...' if include_failed_tasks else ''}
                - Current status of examples of technical issues and solutions for the proposed next task will be tested on: ...
                You should tell me the next best function we should try to implement given your LLM knowledge, available learnt tasks, in order to solve problems to produce it.

                Here you can find a structure of an existing Fiche of Anomalie (FA):
                {structure_FA}

                Cypher request which was used to create the database:
                {cypher_request}

                
            CONSTRAINTS:
                You must follow the criteria below:
                {coach_text}
                 You should propose a plan to achieve this task by breaking it down as a tree-structure. The plan tree should be of depth {plan_depth}.
                Some specific commands available later for implementation in python of the task you will provide:
                {available_commands_detail} list of commands...


        You should only respond in the format as described below:

            RESPONSE FORMAT:
                1. Reasoning: Based on the information listed above, do reasoning about what the next task should be and why. Ensure it will minimize the distance to goal.
                2. Task: Next best task to develop.
                3. Specifications: present a tree-like structure of acceptance criteria, best strategies to compare, performance tips to beat a single LLM.
                4. Plan: Tree-structured plan of depth {plan_depth} breaking-down next best task into basic commands
                {test_example}
    """]

    # Write the prompt in the file readed after by the coach
    with open("./prompts/identify_best_task.txt", "w") as f:
            f.write(prompt_coach[0])

    # Define parameters for Coder

    prompt_template = trial.suggest_categorical("prompt_template", ["Extensive", "Minimal"])
    libraries_restriction = trial.suggest_categorical("libraries_restriction", [
        "BeautifulSoap, RegEx, Sklearn, Huggingface, Langchain, Voyager",
        "Numpy, Pandas, Scikit-learn",
        "TensorFlow, PyTorch, Transformers, SpaCy"
    ])
    bot_function_specifications = trial.suggest_categorical("bot_function_specifications", [
        "Manipulate document sections: bot.create_and_add_section_then_return_id(title: str, content: str, section_id: int = None, parent_id: int = None) -> int, bot.get_all_sections() -> List[Section], bot.get_sections(ids: List[int]) -> List[Section], bot.edit_section(section_id: int, new_content: str = None, new_title: str = None, new_parent_id: int = None) -> bool, bot.remove_section(section_id: int) -> bool, bot.swap_sections(section_id_1: int, section_id_2: int) -> bool",
        "Manipulate document resources: bot.add_or_update_results_in_resources(results, metadatas_to_add:dict=None, store_linked_document_content:bool=False), bot.add_or_update_result_in_resources(metadatas:dict, name:str=None, content:dict=None, link:str=None, store_linked_document_content:bool=False), bot.get_all_resources(self) -> List[Dict[str, Any]], bot.semantic_search_resources(query_texts, n_results=10), bot.add_or_update_results_in_resources(results, metadatas:dict=None, store_linked_document_content:bool=False), bot.get_and_store_link_content(link:str=None, parent_id=None, chaining:bool=True), bot.remove_resource(resource_id)"
    ])
    reasoning_depth = trial.suggest_int("reasoning_depth", 1, 3)
    modularity = trial.suggest_categorical("modularity", [
        "None",
        "Helper functions for common tasks",
        "Modular classes for related functions",
        "Parameterize all varying data"
    ])
    handle_previous_attempts = trial.suggest_categorical("handle_previous_attempts", [True, False])
    instruction_to_remove = trial.suggest_int("instruction_to_remove", 1, 14)

    modularity_final = f"15) Ensure that the generated code adheres to principles of reusability and modularity as {modularity}." if modularity != "None" else "3) Ensure that the generated code adheres to principles of reusability." 


    # Construct the prompt based on the suggested parameters
    criteria_coder = [
        f"1) First part of the function should be dedicated for the Neo4J graph connection (which is already created). Here you can find the informations:",
            f"URI='bolt://127.0.0.1:7687'",
            f"AUTH = auth=('neo4j','password')",
        f"2) Ensure that the generated code adheres to principles of reusability and modularity. Specifically, functions should not hard-code strings, variables, or parameters that make them context-specific. Instead, any data or parameters of helpers functions that can vary should be passed as arguments to the functions, ensuring that the functions can be reused in different contexts or with different data without requiring modifications to the code itself. This ensures that the code is adaptable and can be utilized in various scenarios, enhancing its utility and longevity.",
        f"3) Call existing functions as much as possible from the primitives folder.",
        f"4) Name your function in a meaningful way (can infer the task from the name).",
        f"5) Your function will be reused for building more complex functions. Therefore, you should make it generic and reusable. Avoid to include specific query or information in the function instead of using it as an argument.",
        f"6) Anything defined outside a function will be ignored, define all your variables and classes inside your functions.",
        f"7) Ensure that your code is fully executable, it is not a skeleton and does not contain placeholders, unimplemented sections, or comments indicating future work (e.g., TODO, pass, '....', etc.). All functions and logic must be complete and runnable to facilitate immediate use and testing.",
        f"8) Do not write infinite loops or recursive functions. Do not use Mathplotlib and clustering methods. Find the clothest ways on the graph.",
        f"9) Name your function in a meaningful way (can infer the task from the name).",
        f"10) Any packages/libraries used by the function should be imported inside the function (it will be ignored if imported outside).",
        f"11) dont try to create new tab (columns and rows), or time_range (e.g 'previous 30 days' or WHERE fa.fan_date_declaration >= date('time_range'))",
        f"12) Do not take into account the fan_date_declaration, it does not matter for the code.",
        f"13) Adapt the exact same code as the primitives  (generateNeo4J.py) is their is nothing in the succeded functions",
        f"14) Avoid those errors:"
            "- Error: 'SynthesisManager' object has no attribute 'min_plan_cosine_similarity'"
            "- analyze_similar_anomalies() got an unexpected keyword argument 'problem'",
        {modularity_final}
    ]
    
    del criteria_coder[instruction_to_remove - 1]
    for i in range(len(criteria_coder)):
        coder_text = "\n".join(criteria_coder[i])
    
    prompt_coder = [f"""
    You are a helpful assistant that writes Python code to be executed using a restricted list of packages ({libraries_restriction}) to complete the task specified by me.
    At each round of conversation, I will give you:
    - Reasoning: explanation of the task chosen...
    - Task: ...
    - Plan: ...
    - Tests: tests that will be done on target document

    CURRENT STATE OF THE ENVIRONMENT USED TO TEST TASK
        Document #xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx: 
        {
                'id: ...'
                'title: ...'
                'abstract: ...'
                'comment: ...'
                'target_file_path: ...'
        }

    - General code for re-use or demonstration purpose: ...
    - Code from the last round with attempts to implement task with its performance (e.g. 'sections titles progress': x, 'sections content progress': y): ...
    - Execution error: ...

    Here you can find a structure of an existing Fiche of Anomalie (FA):
        {structure_FA}

    Cypher request which was used to create the database: 
        {cypher_request}
               
    You should then respond to me with:
        - Reasoning: How to best implement the plan with no errors and maximum performance towards the goal ?
        - Code:
        {coder_text}

    RESPONSE FORMAT (You should only respond in the format as described below, and follow the example provided):
        ```python    
                main function after the helper functions
                def your_main_function_name(args...):
                # detailed content of the function...
        ``` 

        Tests: 
        # document #72dc469b-63f8-4751-aab5-6db3d16fca3c usage test:
        your_main_function_name(args specific to document #72dc469b-63f8-4751-aab5-6db3d16fca3c...)

        # document #c0533337-1c5e-4091-ba7d-061ae409cda4 usage test:
        your_main_function_name(args specific to document #c0533337-1c5e-4091-ba7d-061ae409cda4...)
    """]

    # Write the prompt in the file readed after by the coder
    with open("./prompts/code_task.txt", "w") as f:
            f.write(prompt_coder[0])

    # Define parameters for the Critic

    prompt_template = trial.suggest_categorical("prompt_template", ["Extensive", "Minimal"])
    reasoning_depth = trial.suggest_int("reasoning_depth", 1, 3)
    detailed_explanation = trial.suggest_categorical("detailed_explanation", [True, False])
    critic_to_remove = trial.suggest_int("critic_to_remove", 1, 3)
    
    # Construct the prompt based on the suggested parameters
    criteria_critic = [
        f"Reasoning: Based on the information I listed above, do a {reasoning_depth} step reasoning to evaluate if the code implementation and execution is aligned with the task goal to decide if it is a success.",
        "Success: write 'True' if code is a success, 'False' otherwise",
        f"Explain: explain in detail your evaluation of your success evaluation." if detailed_explanation else ""
    ]
    
    del criteria_critic[critic_to_remove - 1]
    critic_text = "\n".join(criteria_critic)
    
    if prompt_template == "Extensive":
        prompt_critic = f"""
        You are a Python expert and domain expert in the field of the task, you should validate the Python code provided and its result regarding the code implementing the task and feedback.
        I will provide you:
        TASK: {{task}}
        CODE: {{code}}
        Some additional information to evaluate code: {{runtime_errors}}
        Execution result returned by exec command of code provided: {{exec_result}}
        RESPONSE FORMAT: you should only respond in the format as described below:
        {critic_text}
        EXAMPLES:
        Reasoning: The initial task was to list all GPS points of vessels in the zone. The code is aligned with this task, it ran without errors, your confirmed what I could not check.
        Success: "True"
        Explain: code generated result expected by task without errors.
        """
    else:
        prompt_critic = f"""
        You are a Python expert and domain expert in the field of the task, you should validate the Python code provided and its result regarding the code implementing the task and feedback.
        I will provide you:
        TASK: {{task}}
        CODE: {{code}}
        Some additional information to evaluate code: {{runtime_errors}}
        Execution result returned by exec command of code provided: {{exec_result}}
        RESPONSE FORMAT: you should only respond in the format as described below:
        {critic_text}
        """

        with open("./prompts/validate_code.txt", "w") as f:
            f.write(prompt_critic)

    # Save the parameters chosen by the trial
    with open("Optuna_results.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nPrompt Coach Chosen: \n{prompt_coach}\nPrompt Coder Chosen : \n{prompt_coder}\nPrompt Critic Chosen : \n{prompt_critic}\nModel Chosen: {modelVariation}\n")

    # Run the learning loop
    perf = learn.run_4agents_learning_loop(default_llm_key="default_llm",
                                premium_llm_key="premium_llm",
                                llmORchains_list=llmORchains_list,
                                test_environments=envs,
                                manual_validation_to_capitalize=False,
                                problem_prompts_subdir='IR_CPS_TechSynthesis', 
                                max_coding_attempts=1,
                                include_code=False,
                                selected_successful_functions=[],
                                selected_failed_functions=[],
                                agtask_premium_llm_by_default=True,
                                agtask_skip_rounds=0,
                                agcoding_skip_rounds=0,
                                agvalidation_skip_rounds=0,
                                agcapitalize_skip_rounds=0,
                                model_choice=modelVariation,
                                optuna_opti="Coach")
    
    with open("Optuna_results.txt", "a") as f:
        f.write(f"Performance: {perf}\n\n")
    
    return perf

if __name__ == "__main__":
    # Initialize the default and premium LLMs
    #default_llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    #default_llm = create_Nmajority_chain(num_models=3)
    #premium_llm = ChatOpenAI(model_name="gpt-4o") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["gpt-3.5"]),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["gpt-4"]),
        #"3_majority_chain": learn.create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["gpt-3.5"], reduce_model_name=MODELS_CONFIG_LIST["gpt-3.5"] , num_models=3),
        #"10_majority_chain": learn.create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["gpt-3.5"], reduce_model_name=MODELS_CONFIG_LIST["gpt-3.5"], num_models=10)
    }

    # Set the documents to test/validate as a list of environments
    documents=[
        {
        "id": "68802377-e8e5-4940-a337-e66930ba5015",
        "title": "Erreur de connexion au serveur interne",
        "context": "Lors de la tentative de connexion au serveur interne de l'entreprise, les utilisateurs rencontrent un message d'erreur indiquant une impossibilité de se connecter. Ce problème semble intermittent et affecte principalement les utilisateurs du département des ventes.",
        "target_file_path": "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/fiche_ano/Erreur de connexion au serveur interne.json"
        },

        {
        "id": "c0533337-1c5e-4091-ba7d-061ae409cda4",
        "title": "Problème de performance sur le module de gestion des utilisateurs",
        "context": "Lors de l'utilisation du module de gestion des utilisateurs, nous avons constaté des ralentissements significatifs. Les utilisateurs rapportent que la page met plusieurs minutes à se charger et que les opérations de modification et de suppression d'utilisateur prennent un temps anormalement long. Ce problème a été observé sur plusieurs navigateurs et sur différentes configurations matérielles, ce qui suggère qu'il ne s'agit pas d'un problème isolé à un utilisateur spécifique ou à un type de machine. Nous avons identifié que ce problème semble se produire principalement lorsque le nombre d'utilisateurs dépasse les 1000. Les logs du serveur montrent des temps de réponse élevés sur les requêtes liées à la base de données pour ce module spécifique. Une analyse initiale indique que certaines requêtes ne sont pas optimisées et causent des verrous au niveau de la base de données.",
        "target_file_path": "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/fiche_ano/Problème de performance sur le module de gestion des utilisateurs.json"
        }
    ]

    envs = []
    for doc in documents:
        env = learn.EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'], target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)

    with open("Optuna_results.txt", "w") as f:
        f.write("")
    # Wait for 10s
    time.sleep(10)
    # get current folder
    current_folder = os.getcwd()

    sqlite_file = os.path.join(current_folder, "optuna.db")

    # Create a study and optimize the objective function
    study = opt.create_study(direction="maximize", storage=f"sqlite:///{sqlite_file}")
    study.optimize(objective, n_trials=200)