import os
# If you use OPENAI, you can add your API key here
OPENAI_API_KEY = "Your_openai_api_key_here"
openai_api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Elastic search and Kibana information
elastic_url_port = 'Your_elastic_url_here' # Default port is 9200
kibana_url_port = 'Your_kibana_url_here' # Default port is 5601
ELASTICSEARCH_HOST = 'http://localhost:9200'
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