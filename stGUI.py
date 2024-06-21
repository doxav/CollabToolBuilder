import streamlit as st
from learn import run_4agents_learning_loop, create_Nmajority_chain, EnvironmentManager
from langchain.chat_models import ChatOpenAI

def main():
    st.title("Learning Loop Interface")

    if 'default_llm_key' not in st.session_state:
        st.session_state['default_llm_key'] = 'default_llm'
    if 'premium_llm_key' not in st.session_state:
        st.session_state['premium_llm_key'] = 'premium_llm'
    if 'max_coding_attempts' not in st.session_state:
        st.session_state['max_coding_attempts'] = 4
    if 'manual_validation' not in st.session_state:
        st.session_state['manual_validation'] = False
    if 'problem_prompts_subdir' not in st.session_state:
        st.session_state['problem_prompts_subdir'] = 'IR_CPS_TechSynthesis'

    default_llm_key = st.selectbox('Select Default LLM', ['default_llm', '3_majority_chain', '10_majority_chain'], key='default_llm_key')
    premium_llm_key = st.selectbox('Select Premium LLM', ['premium_llm', None], key='premium_llm_key')
    max_coding_attempts = st.number_input('Max Coding Attempts', min_value=1, max_value=10, value=4, key='max_coding_attempts')
    manual_validation = st.checkbox('Manual Validation', value=True, key='manual_validation')
    problem_prompts_subdir = st.text_input('Problem Prompts Subdirectory', value='', key='problem_prompts_subdir')

    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name="gpt-3.5-turbo-1106"),
        "premium_llm": ChatOpenAI(model_name="gpt-4o"),
        "3_majority_chain": create_Nmajority_chain(num_models=3),
        "10_majority_chain": create_Nmajority_chain(num_models=10)
    }

    # Set the documents to test/validate as a list of environments
    documents = [{'id': "cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                  'title': "Complex QA and language models hybrid architectures, Survey",
                  'context': "This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
                 {'id': "42252c6c-12f3-4edf-9045-8acd69bc3356",
                  'title': "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
                  'context': "This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]

    envs = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
                                 target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)

    if st.button('Start Learning Loop'):
        run_4agents_learning_loop(
            default_llm_key=st.session_state['default_llm_key'],
            premium_llm_key=st.session_state['premium_llm_key'],
            llmORchains_list=llmORchains_list,
            test_environments=envs,
            manual_validation_to_capitalize=st.session_state['manual_validation'],
            problem_prompts_subdir=st.session_state['problem_prompts_subdir'],
            max_coding_attempts=st.session_state['max_coding_attempts'],
            include_code=False,
            selected_successful_functions=[],
            selected_failed_functions=[],
            agtask_premium_llm_by_default=False,
            agtask_skip_rounds=0,
            agcoding_skip_rounds=0,
            agvalidation_skip_rounds=0,
            agcapitalize_skip_rounds=0)

if __name__ == "__main__":
    main()
