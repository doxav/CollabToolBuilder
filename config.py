import os

os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'ollama'
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai_api_key = OPENAI_API_KEY
elastic_url_port = 'https://compute-lsis-2.lis-lab.fr/elastic'
kibana_url_port = 'https://compute-lsis-2.lis-lab.fr/kibana'
elastic_user="daull"
elastic_password="1OtNitubotjil<"
PickleCacheActivated = False

MODELS_CONFIG_LIST = {
    "gpt-4":"gemma2:9b",
    "gemma":"gemma2:9b",
    "gpt-3.5":"phi3:medium",
}
