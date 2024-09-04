import os

#os.environ['OPENAI_BASE_URL'] = 'http://127.0.0.1:11434' #'ollama'
#os.environ['OPENAI_BASE_URL'] = "https://api.groq.com/openai/v1" #'groq'
#os.environ['OPENAI_API_KEY'] = "sk-am7886MazT0utPS4PWiV4ykxhivvEbIC0yP8J_1cEKT3BlbkFJAoV6uJFBQMy8dkPFNpT_X9BCFCHJ8-bG9CIbtx_oEA" 
#os.environ['OPENAI_API_KEY'] = "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" # Groq
#os.environ["GROQ_API_KEY"] = os.environ['OPENAI_API_KEY']

# Google Gemini API Key: AIzaSyBXJF0CjEOo9dEzZs-Kutgds0Lv34JFqgo

#OPENAI_API_KEY = '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'
OPENAI_API_KEY = "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"
openai_api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

elastic_url_port = 'https://compute-lsis-2.lis-lab.fr/elastic'
kibana_url_port = 'https://compute-lsis-2.lis-lab.fr/kibana'
#elastic_url_port = 'http://127.0.0.1:9200' # not used in our case
#kibana_url_port = 'http://127.0.0.1:5601'
elastic_user="richard"
elastic_password="PF2ng2Kg2&#'WU^"
PickleCacheActivated = False

NEO4J_URI = 'neo4j+s://4fb62461.databases.neo4j.io:7687'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'YM3MtKtDTOPq4CPViqKeqYZhORVAfBAYM-ka0i9Ap2w'
ELASTICSEARCH_HOST = 'http://localhost:9200'

MODELS_CONFIG_LIST = {
    "premium_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "basic_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini" # "eramax/nxcode-cq-7b-orpo:q6" #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
}