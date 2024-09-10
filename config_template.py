import os

#os.environ['OPENAI_BASE_URL'] = 'http://127.0.0.1:11434' #'ollama'


###### Use of Groq API
#os.environ['OPENAI_BASE_URL'] = "https://api.groq.com/openai/v1" #'groq'
#os.environ['OPENAI_API_KEY'] = "your-key-here" 
#os.environ["GROQ_API_KEY"] = os.environ['OPENAI_API_KEY']

OPENAI_API_KEY = "your_key-here"
openai_api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

#elastic_url_port = 'https://compute-lsis-2.lis-lab.fr/elastic'
#kibana_url_port = 'https://compute-lsis-2.lis-lab.fr/kibana'
elastic_url_port = 'http://127.0.0.1:9200' # not used in our case
kibana_url_port = 'http://127.0.0.1:5601'
elastic_user=None
elastic_password=None
PickleCacheActivated = False

NEO4J_URI = 'neo4j+s://4fb62461.databases.neo4j.io:7687'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'your-neo4j-password-here'
ELASTICSEARCH_HOST = 'http://localhost:9200'

MODELS_CONFIG_LIST = {
    "premium_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "smart_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "basic_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini" # "eramax/nxcode-cq-7b-orpo:q6" #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
}