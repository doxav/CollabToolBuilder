import os
import subprocess
from datetime import datetime
from elasticsearch import Elasticsearch
from new_environment import stop_containers
from optimisation.optuna_main import global_main, launch_run
import json
import ast
import time
from elasticsearch.helpers import bulk
from elasticsearch import ConnectionError
import shutil

import sys

# Generate a timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name_exp = f"{os.path.splitext(os.path.basename(__file__))[0]}_{timestamp}"
log_filename = f"logs/{name_exp}.log"

# Redirect stdout to both console and log file
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        if not os.path.exists('logs'):
            os.chdir('..')
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  pass # For compatibility with `sys.stdout`

sys.stdout = DualLogger(log_filename)

def get_docker_compose_command():
    """
    Détermine si 'docker-compose' ou 'docker compose' doit être utilisé.
    """
    if shutil.which('docker-compose'):
        return ['docker-compose']
    elif shutil.which('docker'):
        return ['docker', 'compose']
    else:
        raise FileNotFoundError("Ni 'docker-compose' ni 'docker compose' n'a été trouvé dans le PATH.")

def wait_for_elasticsearch(host, port, timeout=60):
    es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
    start_time = time.time()
    while True:
        try:
            if es.ping():
                print("Elasticsearch est démarré et accessible.")
                return es
            else:
                print("Elasticsearch n'est pas encore accessible.")
        except ConnectionError:
            print("Elasticsearch n'est pas encore accessible.")
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"Impossible de joindre Elasticsearch après {timeout} secondes.")
            return None
        time.sleep(5)

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
                print(f"Timestamp extrait de 'text': {timestamp_dt}")
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
    delays = [3, 6, 9, 18]  # Progressive delays for retries
    for attempt, delay in enumerate(delays, start=1):
        try:
            page = es_client.search(index=es_index, body=query, scroll='2m', size=1000)
            sid = page['_scroll_id']
            all_hits.extend(page['hits']['hits'])

            while len(page['hits']['hits']) > 0:
                page = es_client.scroll(scroll_id=sid, scroll='2m')
                sid = page['_scroll_id']
                all_hits.extend(page['hits']['hits'])

            print(f"Total de documents récupérés dans l'index '{es_index}': {len(all_hits)}")
            return all_hits  # Success; return results if no exceptions

        except Exception as e:
            if attempt == len(delays):
                break  # Last attempt; do not wait
            print(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

    return []  # Return empty list if all retries fail

from datetime import timedelta

def filter_documents_by_timestamp(all_hits, desired_datetime, duration_minutes):
    """
    Filters documents within a desired time range.
    If desired_datetime is None, uses the earliest timestamp found.
    Returns an empty list if duration_minutes is 0 or no valid timestamps exist.
    """
    if duration_minutes == 0:
        return []

    # Extract timestamps and determine desired_datetime if needed
    timestamps = [
        extract_timestamp_from_text(hit['_source'].get('text', '')) or extract_timestamp_from_text(str(hit['_source'].get('metadata', '')))
        for hit in all_hits
    ]
    timestamps = [ts for ts in timestamps if ts]

    if not timestamps:
        return []  # No valid timestamps

    desired_datetime = desired_datetime or min(timestamps)
    end_datetime = desired_datetime + timedelta(minutes=duration_minutes)

    # Filter documents within the time range
    filtered_documents = [
        hit for hit in all_hits
        if desired_datetime <= (timestamp := (extract_timestamp_from_text(hit['_source'].get('text', '')) or extract_timestamp_from_text(str(hit['_source'].get('metadata', ''))))) < end_datetime
    ]

    print(f"{len(filtered_documents)} documents filtered between {desired_datetime} and {end_datetime}")
    return filtered_documents


def copy_backup_to_container(backup_folder : str, container_name : str):
    """
    Copie les fichiers du dossier backup dans le conteneur Elasticsearch sans écraser les fichiers existants.
    """
    if os.path.exists(backup_folder):
        print(f"Copie des fichiers du dossier '{backup_folder}' vers le conteneur '{container_name}'...")
        for root, dirs, files in os.walk(backup_folder):
            for file in files:
                local_path = os.path.join(root, file)
                container_path = os.path.join('/usr/share/elasticsearch/data',
                                              os.path.relpath(local_path, backup_folder))

                # Créer le dossier dans le conteneur si nécessaire
                container_dir = os.path.dirname(container_path)
                subprocess.run(['docker', 'exec', container_name, 'mkdir', '-p', container_dir])
                # Copier le fichier sans écraser
                subprocess.run(['docker', 'cp', local_path, f'{container_name}:{container_path}'])
        print("Copie des fichiers terminée.")
    else:
        print(f"Le dossier '{backup_folder}' n'existe pas.")

def restart_original_container(docker_compose_file):
    """
    Relance le premier conteneur Elasticsearch en utilisant docker compose start.
    """
    print("Redémarrage du conteneur Elasticsearch original...")
    docker_compose_cmd = get_docker_compose_command()
    subprocess.run(docker_compose_cmd + ['-f', docker_compose_file, 'start'])
    print("Conteneur Elasticsearch original redémarré.")

def set_elastic_docker_with_backup_data(docker_compose_file, time_human_xp, name_xp, backup_folder = 'optimisation/backup_'):
    es_host = 'localhost'
    es_port = 9200
    index_pattern = f'*{name_xp}*'   # Filtre pour les noms d'index contenant 'name_exp'

    # Étape 1 : Copie des fichiers du dossier backup dans le conteneur Elasticsearch déjà lancé
    # Obtenir le nom du conteneur Elasticsearch en cours d'exécution
    result = subprocess.run(['docker', 'ps', '--filter', 'ancestor=docker.elastic.co/elasticsearch/elasticsearch:8.10.1', '--format', '{{.Names}}'], capture_output=True, text=True)
    container_name = result.stdout.strip()
    if container_name:
        copy_backup_to_container(backup_folder, container_name)
    else:
        print("Aucun conteneur Elasticsearch en cours d'exécution trouvé.")
        return

    es_client = wait_for_elasticsearch(es_host, es_port, timeout=120)
    if es_client is None:
        print("Arrêt du script en raison de l'indisponibilité de Elasticsearch.")
        return
    # Récupérer la liste des index contenant 'name_exp' dans le nom with retry/sleep mechanism
    delays = [3, 6, 9, 18]
    indices = es_client.indices.get_alias(index=index_pattern)
    for delay in delays:
        try: indices = es_client.indices.get_alias(index=index_pattern); break
        except Exception as e:
            time.sleep(delay) if delay != delays[-1] else print(f"Error: {e}. No more retries.")
    index_names = list(indices.keys())
    if not index_names:
        print(f"Aucun index trouvé avec le motif '{index_pattern}'.")
        return

    all_filtered_documents = []
    for es_index in index_names:
        print(f"\nTraitement de l'index '{es_index}'...")
        # Obtenir le mapping de l'ancien index
        old_mapping = es_client.indices.get_mapping(index=es_index)

        # Étape 2 : Récupérer tous les documents de l'index
        all_hits = get_all_documents(es_client, es_index)

        # Extraire la date et l'heure du nom de l'index
        index_parts = es_index.rsplit('_', 1)
        if len(index_parts) == 2:
            date_time_str = index_parts[1]
            try:
                desired_datetime = datetime.strptime(date_time_str, '%d-%m-%Y-%H-%M-%S')
            except ValueError as e:
                print(f"Error in trying to convert from backup name '{es_index}': {e}\nUse current time instead.")
                desired_datetime = None
        else:
            print(f"Format du nom de l'index inattendu pour '{es_index}'.")
            continue

        duration_minutes = time_human_xp  # Durée en minutes du filtre
        desired_datetime = None

        # Étape 3 : Filtrer les documents en fonction du timestamp extrait de 'text'
        filtered_documents = filter_documents_by_timestamp(all_hits, desired_datetime, duration_minutes)

        # Ajouter les documents filtrés à la liste globale
        for doc in filtered_documents:
            doc['_index'] = es_index  # Assurez-vous de conserver le nom de l'index
            all_filtered_documents.append(doc)

        # Sauvegarder les mappings pour chaque index
        if not os.path.exists('mappings'):
            os.makedirs('mappings')
        with open(f'mappings/{es_index}_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(old_mapping[es_index], f, ensure_ascii=False, indent=4)

    print(f"\nNombre total de documents après filtrage : {len(all_filtered_documents)}")

    # **Étape 4 : Mettre les documents filtrés dans le dossier 'backup'**
    backup_folder = 'backup_temp'
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        print(f"Dossier '{backup_folder}' créé.")

    # Sauvegarder les documents dans un seul fichier JSON
    backup_file = os.path.join(backup_folder, 'filtered_documents.json')
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(all_filtered_documents, f, ensure_ascii=False, indent=4)
    print(f"Les documents filtrés ont été sauvegardés dans {backup_file}")

    # Arrêter le conteneur Elasticsearch original
    print("Arrêt du conteneur Elasticsearch original...")
    docker_compose_cmd = get_docker_compose_command()
    subprocess.run(docker_compose_cmd + ['-f', docker_compose_file, 'stop'])
    print("Conteneur Elasticsearch original arrêté.")

    # Démarrer une nouvelle instance Elasticsearch temporaire
    subprocess.check_call(
        ['python3', 'new_environment.py', '--folder', 'backup', '--docker_compose_file', 'docker-compose-backup.yml'])

    # Attendre que la nouvelle instance Elasticsearch soit disponible
    new_es_client = wait_for_elasticsearch(es_host, es_port, timeout=120)
    if new_es_client is None:
        print("Arrêt du script en raison de l'indisponibilité de la nouvelle instance Elasticsearch.")
        return

    # Créer les index dans la nouvelle instance et insérer les documents
    for es_index in index_names:
        # Charger le mapping sauvegardé
        with open(f'mappings/{es_index}_mapping.json', 'r', encoding='utf-8') as f:
            old_mapping = json.load(f)

        # Créer l'index dans la nouvelle instance Elasticsearch
        if not new_es_client.indices.exists(index=es_index):
            new_es_client.indices.create(index=es_index, body={
                "mappings": old_mapping['mappings']
            })
            print(f"Index '{es_index}' créé dans la nouvelle instance Elasticsearch.")

    # Préparer les actions pour l'API bulk
    actions = []
    for doc in all_filtered_documents:
        action = {
            "_index": doc['_index'],
            "_id": doc['_id'],
            "_source": doc['_source']
        }
        actions.append(action)

    # Insérer les documents dans la nouvelle instance Elasticsearch
    success, _ = bulk(new_es_client, actions)
    print(f"{success} documents ont été insérés dans la nouvelle instance Elasticsearch.")

    return success

initial_HumanXp_duration_values = [50] # [0, 20, 30, 60]
name_experiments = ["auto", "msi_30-10-2024-14-58-34", "msi_30-10-2024-14-25-12", "msi_30-10-2024-15-01-19", "msi_29-10-2024-20-28-27", "msi_29-10-2024-20-27-18", "msi_30-10-2024-13-20-11"] # "msi_30-10-2024-14-25-12" "msi_20-10-2024-12-02-28" "msi_30-10-2024-14-58-34" ["auto"]
name_experiments = ["msi_30-10-2024-14-58-34", "msi_30-10-2024-15-01-19", "msi_29-10-2024-20-28-27", "msi_30-10-2024-13-20-11"] # "auto", "msi_29-10-2024-20-27-18", "msi_30-10-2024-14-25-12"] #
name_experiments = ["msi_29-10-2024-20-27-18"] # "auto", "msi_29-10-2024-20-27-18", "msi_30-10-2024-14-25-12"] #
#totalxp_duration_values = [30, 30, 30]
totalxp_duration_values = [10]
#name_experiments = ["msi_30-10-2024-14-58-34"] # "msi_30-10-2024-14-25-12" "msi_20-10-2024-12-02-28" "msi_30-10-2024-14-58-34"
from optuna.samplers import GridSampler
my_grid = GridSampler({'time_human_xp': initial_HumanXp_duration_values, 'name_xp': name_experiments, 'totalxp_duration_value': totalxp_duration_values})
n_trials = len(initial_HumanXp_duration_values) * len(name_experiments) * len(totalxp_duration_values)
skip_learning_to_test_backup = False
time_xp_auto_deduced_from_human_xp = False

def objective(trial, name_xp : str):
    global initial_HumanXp_duration_values, skip_learning_to_test_backup, time_xp_auto_deduced_from_human_xp, totalxp_duration_values

    time_human_xp = trial.suggest_categorical("time_human_xp", initial_HumanXp_duration_values)
    totalxp_duration_value = trial.suggest_categorical("totalxp_duration_value", totalxp_duration_values)
    if time_xp_auto_deduced_from_human_xp:
        time_xp_auto = max((totalxp_duration_value - time_human_xp), 1) * 60 # To avoid 0 time, minimum 1 minute
    else:
        time_xp_auto = totalxp_duration_value * 60
    print(f"Time_human_xp: {time_human_xp}, Time_xp_auto: {time_xp_auto}")

    name_experiment = trial.suggest_categorical("name_xp", name_experiments)

    number_selected = set_elastic_docker_with_backup_data('docker-compose-backup.yml', time_human_xp, name_experiment)
    performance = 0
    if not skip_learning_to_test_backup:
        performance = launch_run(default_llm_key="default_llm",
                                 premium_llm_key="premium_llm",
                                 problem_prompts_subdir="IR_CPS_TechSynthesis",
                                 max_execution_time=time_xp_auto,
                                 max_coding_attempts=10,
                                 # special_criteria={"CodingAgent#max_autofix": 2},
                                 model_choice={"coach": "default_llm","coder": "premium_llm","critic": "default_llm","capitalizer": "default_llm"},
                                 automation="coach",
                                 name_exp=name_experiment,
                                 arrayn_ret=True,
                                 continue_each_loop=True)

        with open(f'Optuna_results/{name_xp}.txt', 'a') as f:
            f.write(f"Time_human_xp: {time_human_xp}\n")
            f.write(f"performances: {performance}\n")
            if performance and len(performance) > 0:
                f.write(f"performance: {max(performance)}\n")
            else:
                f.write(f"performance: None\n")

    stop_containers('docker-compose-backup.yml')
    restart_original_container('elasticsearch/docker-compose.yml')

    if not skip_learning_to_test_backup:
        return max(performance)
    else:
        return number_selected

if __name__ == "__main__":
    global_main(objective, '', 'backup', name_exp=name_exp, n_trials=n_trials, sampler=my_grid)
