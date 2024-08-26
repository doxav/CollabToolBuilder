import os

#os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = "sk-None-VCFqv8mthO2iMLbSgMoUT3BlbkFJmpneVutMLtU1mZ3FsV9w" #'ollama'
# Google Gemini API Key: AIzaSyBXJF0CjEOo9dEzZs-Kutgds0Lv34JFqgo
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai_api_key = OPENAI_API_KEY
elastic_url_port = 'https://compute-lsis-2.lis-lab.fr/elastic'
kibana_url_port = 'https://compute-lsis-2.lis-lab.fr/kibana'
elastic_url_port = 'http://127.0.0.1:9200'
kibana_url_port = 'http://127.0.0.1:5601'
elastic_user=None
elastic_password=None
PickleCacheActivated = False

MODELS_CONFIG_LIST = {
    "gpt-4": "gpt-4o", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "gemma": "gpt-4o-mini", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "gpt-3.5": "gpt-4o-mini" # "eramax/nxcode-cq-7b-orpo:q6" #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
}
