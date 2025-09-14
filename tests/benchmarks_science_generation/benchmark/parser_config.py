from argparse import ArgumentParser


def costorm_parser():
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/co-storm",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"],
        help="The search engine API to use for retrieving information.",
    )
    # hyperparameters for co-storm
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=10,
        help="Retrieve top k results for each query in retriever.",
    )
    parser.add_argument(
        "--max_search_queries",
        type=int,
        default=2,
        help="Maximum number of search queries to consider for each question.",
    )
    parser.add_argument(
        "--total_conv_turn",
        type=int,
        default=20,
        help="Maximum number of turns in conversation.",
    )
    parser.add_argument(
        "--max_search_thread",
        type=int,
        default=5,
        help="Maximum number of parallel threads for retriever.",
    )
    parser.add_argument(
        "--max_search_queries_per_turn",
        type=int,
        default=3,
        help="Maximum number of search queries to consider in each turn.",
    )
    parser.add_argument(
        "--warmstart_max_num_experts",
        type=int,
        default=3,
        help="Max number of experts in perspective-guided QA during warm start.",
    )
    parser.add_argument(
        "--warmstart_max_turn_per_experts",
        type=int,
        default=2,
        help="Max number of turns per perspective during warm start.",
    )
    parser.add_argument(
        "--warmstart_max_thread",
        type=int,
        default=3,
        help="Max number of threads for parallel perspective-guided QA during warm start.",
    )
    parser.add_argument(
        "--max_thread_num",
        type=int,
        default=10,
        help=(
            "Maximum number of threads to use. "
            "Consider reducing it if you keep getting 'Exceed rate limit' errors when calling the LM API."
        ),
    )
    parser.add_argument(
        "--max_num_round_table_experts",
        type=int,
        default=2,
        help="Max number of active experts in round table discussion.",
    )
    parser.add_argument(
        "--moderator_override_N_consecutive_answering_turn",
        type=int,
        default=3,
        help=(
            "Number of consecutive expert answering turns before the moderator overrides the conversation."
        ),
    )
    parser.add_argument(
        "--node_expansion_trigger_count",
        type=int,
        default=10,
        help="Trigger node expansion for nodes that contain more than N snippets.",
    )

    # Boolean flags
    parser.add_argument(
        "--enable_log_print",
        action="store_true",
        help="If set, enable console log print.",
    )
    return parser


def storm_parser():
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/gpt",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="Maximum number of threads to use. The information seeking part and the article generation"
        "part can speed up by using multiple threads. Consider reducing it if keep getting "
        '"Exceed rate limit" error when calling LM API.',
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=[
            "bing",
            "you",
            "brave",
            "serper",
            "duckduckgo",
            "tavily",
            "searxng",
            "azure_ai_search",
        ],
        help="The search engine API to use for retrieving information.",
    )
    # stage of the pipeline
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="If True, simulate conversation to research the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If True, polish the article by adding a summarization section and (optionally) removing "
        "duplicate content.",
    )
    # hyperparameters for the pre-writing stage
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="Maximum number of questions in conversational question asking.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Maximum number of perspectives to consider in perspective-guided question asking.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="Top k search results to consider for each search query.",
    )
    # hyperparameters for the writing stage
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=3,
        help="Top k collected references for each section title.",
    )
    parser.add_argument(
        "--remove-duplicate",
        action="store_true",
        help="If True, remove duplicate content from the article.",
    )
    return parser

def ai_scientist_parser():
    parser = ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/i_cant_believe_its_not_better.json",
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as ideas file but .py extension",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    return parser

def agent_laboratory_parser():
    parser = ArgumentParser(description="AgentLaboratory Research Workflow")

    parser.add_argument(
        '--copilot-mode',
        type=str,
        default="False",
        help='Enable human interaction mode.'
    )

    parser.add_argument(
        '--deepseek-api-key',
        type=str,
        help='Provide the DeepSeek API key.'
    )

    parser.add_argument(
        '--load-existing',
        type=str,
        default="False",
        help='Do not load existing state; start a new workflow.'
    )

    parser.add_argument(
        '--load-existing-path',
        type=str,
        help='Path to load existing state; start a new workflow, e.g. state_saves/results_interpretation.pkl'
    )

    parser.add_argument(
        '--research-topic',
        type=str,
        help='Specify the research topic.'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Provide the OpenAI API key.'
    )

    parser.add_argument(
        '--compile-latex',
        type=str,
        default="True",
        help='Compile latex into pdf during paper writing phase. Disable if you can not install pdflatex.'
    )

    parser.add_argument(
        '--llm-backend',
        type=str,
        default="o3-mini",
        help='Backend LLM to use for agents in Agent Laboratory.'
    )

    parser.add_argument(
        '--language',
        type=str,
        default="English",
        help='Language to operate Agent Laboratory in.'
    )

    parser.add_argument(
        '--num-papers-lit-review',
        type=str,
        default="5",
        help='Total number of papers to summarize in literature review stage'
    )

    parser.add_argument(
        '--mlesolver-max-steps',
        type=str,
        default="3",
        help='Total number of mle-solver steps'
    )

    parser.add_argument(
        '--papersolver-max-steps',
        type=str,
        default="5",
        help='Total number of paper-solver steps'
    )


    return parser