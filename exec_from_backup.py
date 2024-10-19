import subprocess
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from new_environment import stop_containers
from optimisation.optuna_main import launch_run
import json
import ast

def extract_timestamp_from_text(text_field):
    """
    Extrait le timestamp du champ 'time' dans 'text' et le convertit en objet datetime.
    """
    if not text_field:
        print("Le champ 'text' est vide.")
        return None

    try:
        # Essayer de parser le champ 'text' comme JSON
        text_dict = json.loads(text_field)
    except json.JSONDecodeError:
        try:
            # Si JSON échoue, essayer avec ast.literal_eval
            text_dict = ast.literal_eval(text_field)
        except (ValueError, SyntaxError):
            print("Impossible de parser le champ 'text' en dictionnaire.")
            return None

    if isinstance(text_dict, dict):
        time_str = text_dict.get('time')
        if time_str:
            # Ajuster le format de la date si nécessaire
            try:
                timestamp_dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
                return timestamp_dt
            except ValueError:
                print(f"Format de date invalide pour 'time': {time_str}")
                return None
        else:
            print("La clé 'time' n'est pas présente dans 'text'.")
            return None
    else:
        print("Le champ 'text' n'est pas un dictionnaire après parsing.")
        return None

def get_all_documents(es_client, es_index):
    """
    Récupère tous les documents de l'index spécifié.
    """
    query = {
        "query": {
            "match_all": {}
        }
    }

    all_hits = []
    page = es_client.search(
        index=es_index,
        body=query,
        scroll='2m',
        size=1000
    )
    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])
    all_hits.extend(page['hits']['hits'])

    while scroll_size > 0:
        page = es_client.scroll(scroll_id=sid, scroll='2m')
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])
        all_hits.extend(page['hits']['hits'])

    print(f"Total de documents récupérés : {len(all_hits)}")
    return all_hits

def filter_documents_by_timestamp(all_hits, desired_datetime, duration_minutes):
    """
    Filtre les documents dont le timestamp extrait de 'text' est dans la plage de temps désirée.
    """
    end_datetime = desired_datetime + timedelta(minutes=duration_minutes)
    filtered_documents = []
    for hit in all_hits:
        source = hit['_source']
        text_field = source.get('text', '')
        timestamp_dt = extract_timestamp_from_text(text_field)
        if timestamp_dt:
            if desired_datetime <= timestamp_dt < end_datetime:
                filtered_documents.append(hit)
    print(f"{len(filtered_documents)} documents filtrés entre {desired_datetime} et {end_datetime}")
    return filtered_documents

def main():
    es_host = 'localhost'
    es_port = 9200
    es_index = 'failed_tasks_thomas-precision-3591_18-10-2024-20-45-09'  # Remplacez par le nom de votre index

    es_client = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': 'http'}])

    # Étape 1 : Récupérer tous les documents de l'index
    all_hits = get_all_documents(es_client, es_index)

    # Nouvelle étape : Extraire la date et l'heure du nom de l'index
    index_parts = es_index.rsplit('_', 1)
    if len(index_parts) == 2:
        date_time_str = index_parts[1]  # '18-10-2024-20-45-09'
        try:
            desired_datetime = datetime.strptime(date_time_str, '%d-%m-%Y-%H-%M-%S')
        except ValueError as e:
            print(f"Erreur lors de la conversion de la date et l'heure : {e}")
            return
    else:
        print("Format du nom de l'index inattendu.")
        return

    duration_minutes = 60  # Durée en minutes du filtre

    # Étape 3 : Filtrer les documents en fonction du timestamp extrait de 'text'
    filtered_documents = filter_documents_by_timestamp(all_hits, desired_datetime, duration_minutes)

    # Étape 4 : Effectuer les calculs sur les documents filtrés
    print(f"Nombre de documents après filtrage : {len(filtered_documents)}")

    # Exécuter le processus automatique pendant X secondes
    run_duration = 1200  # Durée en secondes du processus automatique

    # Stop existing containers and start new ones from backup
    subprocess.check_call(
        ['python3', 'new_environment.py', '--folder', 'backup', '--docker_compose_file', 'docker-compose-backup.yml'])

    launch_run(max_execution_time=run_duration,
               problem_prompts_subdir='IR_CPS_TechSynthesis',
               default_llm_key='basic_gpt',
               premium_llm_key='smart_gpt')

    print("Processus automatique terminé.")
    stop_containers('docker-compose-backup.yml')

if __name__ == "__main__":
    main()
