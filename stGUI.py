import os
import subprocess
import streamlit as st
import time
import threading
import asyncio
import websockets
from queue import Queue, Empty
from langchain_openai import ChatOpenAI
from learn import create_Nmajority_chain, EnvironmentManager, run_4agents_learning_loop
import utils.llm_utils as llm_utils

# Initialiser l'état de session pour stocker les données
if 'page1_data' not in st.session_state:
    st.session_state['page1_data'] = []

if 'page2_data' not in st.session_state:
    st.session_state['page2_data'] = []

if 'page3_data' not in st.session_state:
    st.session_state['page3_data'] = []

if 'page4_data' not in st.session_state:
    st.session_state['page4_data'] = []

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

if 'thread_started' not in st.session_state:
    st.session_state['thread_started'] = False

# Vérifier si le mode large est activé dans session_state
if "wide_mode" not in st.session_state:
    st.session_state.wide_mode = False

if st.session_state.wide_mode:
    st.set_page_config(page_title="CollabFunctionsGPTCreator", layout="wide")
else:
    st.set_page_config(page_title="CollabFunctionsGPTCreator", layout="centered")

print('Initialisation de l\'état de session terminée.')

llm_utils.ON_INPUT = False


@st.cache_resource
def get_message_queue():
    mq = Queue()
    for _ in range(5):
        mq.put("Pipo")
        print("Pipo added to the queue - queue size: ", mq.qsize())
    return mq


@st.cache_resource
def init_websocket():
    return websockets.connect("ws://localhost:6789")


# check if message_queue exists in global
#if True or 'message_queue' not in globals() and 'data_queue' not in st.session_state:
message_queue = get_message_queue()
print("Queue storage is set to : " + (
    "st.session_state['data_queue']" if 'data_queue' in st.session_state else "message_queue"))

ws_url = "ws://localhost:6789"


async def send_message(message):
    print(f"Websockets Envoi du message : {message}")
    async with websockets.connect(ws_url) as websocket:
        await websocket.send(message)


async def websocket_receive():
    async with websockets.connect(ws_url) as websocket:
        while True:
            message = await websocket.recv()
            (st.session_state['data_queue'] if 'data_queue' in st.session_state else message_queue).put(message)
            print(
                f"Websockets réception du message et ajouté à la queue : {message} - queue size: {(st.session_state['data_queue'] if 'data_queue' in st.session_state else message_queue).qsize()}")


def websocket_receive_thread():
    asyncio.run(websocket_receive())


def process_messages():
    print("Début du traitement des messages")
    while not (st.session_state['data_queue'] if 'data_queue' in st.session_state else message_queue).empty():
        print('queue size:',
              (st.session_state['data_queue'] if 'data_queue' in st.session_state else message_queue).qsize())
        try:
            message = (st.session_state[
                           'data_queue'] if 'data_queue' in st.session_state else message_queue).get()  # Utilisez get() au lieu de get_nowait()
            print(f"Message reçu : {message}")
            st.session_state['page1_data'].append(message)
        except Empty:
            print("La file d'attente est vide.")
            break
    print("Fin du traitement des messages")


# Fonction pour ajouter périodiquement des éléments à la file d'attente
def add_item_periodically():
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name="gpt-3.5-turbo-1106"),
        "premium_llm": ChatOpenAI(model_name="gpt-4o"),
        "3_majority_chain": create_Nmajority_chain(num_models=3),
        "10_majority_chain": create_Nmajority_chain(num_models=10)
    }

    documents = [{'id': "cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                  'title': "Complex QA and language models hybrid architectures, Survey",
                  'context': "This paper reviews the state-of-the-art of language models architectures and strategies "
                             "for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large "
                             "Language Models (LLM) are good at leveraging public data on standard problems but once "
                             "you want to tackle more specific complex questions or problems (e.g. How does the "
                             "concept of personal freedom vary between different cultures ? What is the best mix of "
                             "power generation methods to reduce climate change ?) you may need specific "
                             "architecture, knowledge, skills, methods, sensitive data protection, explainability, "
                             "human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA "
                             "have allowed non-specialists to grasp the great potential as well as the equally strong "
                             "limitations of LLM in complex QA. In this paper, we start by reviewing required skills "
                             "and evaluation techniques. We integrate findings from the robust community edited "
                             "research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and "
                             "challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. "
                             "fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges "
                             "associated with complex QA, including domain adaptation, decomposition and efficient "
                             "multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data "
                             "protection, multimodal search, hallucinations, explainability and truthfulness, "
                             "temporal reasoning. We analyze current solutions and promising research trends, "
                             "using elements such as: hybrid LLM architectural patterns, training and prompting "
                             "strategies, active human reinforcement learning supervised with AI, neuro-symbolic and "
                             "structured knowledge grounding, program synthesis, iterated decomposition and others.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA "
                                      "and language models hybrid architectures Survey.json"},
                 {'id': "42252c6c-12f3-4edf-9045-8acd69bc3356",
                  'title': "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
                  'context': "This paper surveys the empirical literature of inflation targeting. The main findings "
                             "from our review are the following: there is robust empirical evidence that larger and "
                             "more developed countries are more likely to adopt the IT regime; the introduction of "
                             "this regime is conditional on previous disinflation, greater exchange rate flexibility, "
                             "central bank independence, and higher level of financial development; the empirical "
                             "evidence has failed to provide convincing evidence that IT itself may serve as an "
                             "effective tool for stabilizing inflation expectations and for reducing inflation "
                             "persistence; the empirical research focused on advanced economies has failed to provide "
                             "convincing evidence on the beneficial effects of IT on inflation performance, "
                             "while there is some evidence that the gains from the IT regime may have been more "
                             "prevalent in the emerging market economies; there is not convincing evidence that IT is "
                             "associated with either higher output growth or lower output variability; the empirical "
                             "research suggests that IT may have differential effects on exchange-rate volatility in "
                             "advanced economies versus EMEs; although the empirical evidence on the impact of IT on "
                             "fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal "
                             "discipline; the empirical support to the proposition that IT is associated with lower "
                             "disinflation costs seems to be rather weak. Theref    ore, the accumulated empirical "
                             "literature implies that IT does not produce superior macroeconomic benefits in "
                             "comparison with the alternative monetary strategies or, at most, they are quite modest.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv"
                                      "/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  "
                                      "Literature.json"}]
    envs = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
                                 target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)

    print("Starting learning loop...")
    run_4agents_learning_loop(default_llm_key="default_llm",
                              premium_llm_key="default_llm",
                              llmORchains_list=llmORchains_list,
                              test_environments=envs,
                              manual_validation_to_capitalize=False,
                              problem_prompts_subdir='IR_CPS_TechSynthesis',
                              max_coding_attempts=4,
                              include_code=False,
                              selected_successful_functions=[],
                              selected_failed_functions=[],
                              agtask_premium_llm_by_default=False,
                              agtask_skip_rounds=0,  # Auto-test: 1
                              agcoding_skip_rounds=0,  # Auto-test: 4
                              agvalidation_skip_rounds=0,  # Auto-test: 4
                              agcapitalize_skip_rounds=0)  # Auto-test: 0


if st.session_state['thread_started'] is False:
    threading.Thread(target=add_item_periodically, daemon=True).start()
    threading.Thread(target=websocket_receive_thread, daemon=True).start()
    st.session_state['thread_started'] = True


def write_input_data(new_item):
    print("Writing input data")
    asyncio.run(send_message(new_item))
    st.session_state['ON_INPUT'] = False
    st.experimental_rerun()


def change_page(page_name):
    st.session_state['current_page'] = page_name
    st.rerun()


def show_page():
    if st.session_state['current_page'] == 'home':
        show_home_page()
    elif st.session_state['current_page'] == 'page1':
        show_page1()
    elif st.session_state['current_page'] == 'page2':
        show_page2()
    elif st.session_state['current_page'] == 'page3':
        show_page3()
    elif st.session_state['current_page'] == 'page4':
        show_page4()


def show_home_page():
    st.title('Home Page')
    st.write("Welcome to CollabFunctionsGPTCreator's project! Please select a page to navigate.")
    col11, col12, col13, col14 = st.columns(4)
    with col11:
        if st.button("Learning Loop", help='Click to start the learning loop.'):
            change_page('page1')
    with col12:
        if st.button('Chatbot AI', help='Click to chat with the AI.'):
            change_page('page2')


def show_page1():
    global ON_INPUT
    st.title('CollabFunctionsGPTCreator')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('Back to Home Page', help='Click to go back to the home page.'):
            change_page('home')

    with st.sidebar:
        st.write('Settings')
        if st.toggle("Wide Mode"):
            toggle_wide_mode()
        st.number_input('Generation temperature', min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        st.radio('Select model', ['default_llm', 'premium_llm'], index=0)
        st.selectbox('Choose default LLM', ['default_llm', '3_majority_chain', '10_majority_chain'])
        st.selectbox('Choose premium LLM', ['premium_llm', None])

    with st.expander(label=time.strftime("%H:%M:%S", time.localtime()) + " - Task Agent", expanded=True, icon='🤖'):
        col1, col2, col3 = st.columns(3)
        with col3:
            st.button('Open in VSCode', help='Click to open the file in VSCode.')

        placeholder = st.empty()

    concatenated_code = '\n'.join(st.session_state['page1_data'])
    placeholder.code(concatenated_code, language='python')

    print('ON_INPUT:', llm_utils.ON_INPUT)
    print('Queue status:',
          (st.session_state['data_queue'] if 'data_queue' in st.session_state else message_queue).qsize())

    while not llm_utils.ON_INPUT:
        process_messages()
        text = ""
        for message in st.session_state['page1_data']:
            text += message + '\n'
            placeholder.code(text, language='python')
            time.sleep(0.1)
        st.session_state['page1_data'] = []
        time.sleep(1)

    if llm_utils.ON_INPUT:
        time.sleep(1)
        process_messages()
        text = ""
        for message in st.session_state['page1_data']:
            text += message + '\n'
            placeholder.code(text, language='python')
            time.sleep(0.1)
        st.session_state['page1_data'] = []
        with st.form(key='my_form'):
            if llm_utils.AGENT is None:
                choice = st.selectbox('Choose an action',
                                      ["Modify agent's system prompt (role, global context, constraints, examples).",
                                       "Add instruction or information to agent.",
                                       "Skip and set LLM output from recent outputs or manually define it.",
                                       "Log comments (not used by model, just for information).",
                                       "See all previous results for this agent.",
                                       "See previous MODIFIED/SCORED/COMMENTED results for this agent.",
                                       "Skip human actions for N rounds.",
                                       "Change default LLM.",
                                       "Change premium LLM.",
                                       "Change num of parallel inferences.",
                                       "Exit program.",
                                       "Proceed to inference using a PREMIUM LLM."], key="choice")
                new_item = ''
                if choice == "Modify agent's system prompt (role, global context, constraints, examples).":
                    new_item = 'A'
                elif choice == "Add instruction or information to agent.":
                    new_item = 'B'
                elif choice == "Skip and set LLM output from recent outputs or manually define it.":
                    new_item = 'C'
                elif choice == "Log comments (not used by model, just for information).":
                    new_item = 'D'
                elif choice == "See all previous results for this agent.":
                    new_item = 'E'
                elif choice == "See previous MODIFIED/SCORED/COMMENTED results for this agent.":
                    new_item = 'F'
                elif choice == "Skip human actions for N rounds.":
                    new_item = 'G'
                elif choice == "Change default LLM.":
                    new_item = 'H'
                elif choice == "Change premium LLM.":
                    new_item = 'I'
                elif choice == "Change num of parallel inferences.":
                    new_item = 'J'
                elif choice == "Exit program.":
                    new_item = 'K'
                elif choice == "Proceed to inference using a PREMIUM LLM.":
                    new_item = 'P'
            else:
                print(f"Agent: {llm_utils.AGENT}")

            send_button = st.form_submit_button('Send', on_click=write_input_data, args=(new_item,))


def show_page2():
    st.title('Page 2')
    new_text = st.text_area('Saisir du texte pour la Page 2')
    if st.button('Ajouter le texte'):
        if new_text:
            st.session_state['page2_data'].append(new_text)
    st.write('Texte actuel :')
    st.write(' '.join(st.session_state['page2_data']))
    if st.button('Retour à l\'accueil'):
        change_page('home')


def show_page3():
    st.title('Page 3')
    new_item = st.text_input('Add an item to Page 3')
    if st.button('Ajouter'):
        if new_item:
            st.session_state['page3_data'].append(new_item)
    st.write(st.session_state['page3_data'])
    if st.button('Back to Home Page'):
        change_page('home')


def show_page4():
    st.title('Page 4')
    new_text = st.text_area('Saisir du texte pour la Page 4')
    if st.button('Ajouter le texte'):
        if new_text:
            st.session_state['page4_data'].append(new_text)
    st.write('Texte actuel :')
    st.write(' '.join(st.session_state['page4_data']))
    if st.button('Retour à l\'accueil'):
        change_page('home')


def open_vscode(file_path, line_number):
    if os.path.exists(file_path):
        subprocess.run(["code", "-g", f"{file_path}:{line_number}"])
    else:
        st.error("Le fichier n'existe pas.")


def toggle_wide_mode():
    st.session_state.wide_mode = not st.session_state.wide_mode
    st.experimental_rerun()


col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
with col8:
    st.link_button(':grey_question:', "https://www.example.com/")

# Afficher la page
show_page()
