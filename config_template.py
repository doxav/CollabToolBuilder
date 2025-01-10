import os

unique_id = None # e.g. set it to "XP_21_12_24" or any appropriate name for comparing on same experiment

# Check the docker environment variables to see if LLM_MODE is set (to launch on a cluster/macstudio using ollama)
IN_MACSTU = os.environ.get('LLM_MODE')
if IN_MACSTU in ["macstudio", "gpu"]:
    os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
    OPENAI_API_KEY = "ollama"
    if IN_MACSTU == "macstudio":
        MODELS_CONFIG_LIST = {
                "basic_gpt":"codestral_latest_20k:latest",
                "smart_gpt":"codestral_latest_20k:latest",
                "code_gpt":"codestral_latest_20k:latest",
        }
    else :
        MODELS_CONFIG_LIST = {
                "basic_gpt":"codestral:latest-20k",
                "smart_gpt":"codestral:latest-20k",
                "code_gpt":"codestral:latest-20k",
        }

else:
    OPENAI_API_KEY = "your_openai_api_key_here"
    MODELS_CONFIG_LIST = {
            "code_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "smart_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "basic_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini" # "eramax/nxcode-cq-7b-orpo:q6" #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    }

openai_api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Elastic search and Kibana information
elastic_url_port = 'Your_elastic_url_here' # Default port is 9200
kibana_url_port = 'Your_kibana_url_here' # Default port is 5601

# If there is a user and password needed for the elastic search, add them here
elastic_user="username"
elastic_password="password"
PickleCacheActivated = False

# Neo4j database information
NEO4J_URI = 'URI_of_your_neo4j_database'
NEO4J_USER = 'neo4j_user'
NEO4J_PASSWORD = 'neo4j_password'

# List of models to use in the application, you can add your own models here, no limitation, for example:
MODELS_CONFIG_LIST = {
    "code_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "smart_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "basic_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini" # "eramax/nxcode-cq-7b-orpo:q6" #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
}
