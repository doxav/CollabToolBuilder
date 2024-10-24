import os
import subprocess
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from new_environment import stop_containers
from optimisation.optuna_main import global_main, launch_run
import json
import ast
import time
from elasticsearch.helpers import bulk
from elasticsearch import ConnectionError
import shutil

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
        size=500,
        request_timeout=120
    )
    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])
    all_hits.extend(page['hits']['hits'])

    while scroll_size > 0:
        page = es_client.scroll(scroll_id=sid, scroll='2m')
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])
        all_hits.extend(page['hits']['hits'])

    print(f"Total de documents récupérés dans l'index '{es_index}': {len(all_hits)}")
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

                # Vérifier si le fichier existe déjà dans le conteneur
                result = subprocess.run(['docker', 'exec', container_name, 'test', '-f', container_path],
                                        capture_output=True)

                if result.returncode == 0:
                    # Fichier existe déjà, fusionner (ajouter un suffixe pour éviter d'écraser)
                    base, ext = os.path.splitext(file)
                    new_file = f"{base}_fusion{ext}"
                    container_path = os.path.join(container_dir, new_file)
                    print(f"Fichier '{file}' existe déjà, renommé en '{new_file}' pour éviter l'écrasement.")

                # Copier le fichier dans le conteneur
                subprocess.run(['docker', 'cp', local_path, f'{container_name}:{container_path}'])
                print(f"Fichier '{local_path}' copié dans '{container_path}'")

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

def main(docker_compose_file, time_human_xp, name_xp):
    print(os.getcwd())
    backup_folder = 'backup'
    es_host = 'localhost'
    es_port = 9200
    index_pattern = f'*{name_xp}*'   # Filtre pour les noms d'index contenant 'name_exp'

    # Étape 1 : Copie des fichiers du dossier backup dans le conteneur Elasticsearch déjà lancé
    # Obtenir le nom du conteneur Elasticsearch en cours d'exécution
    result = subprocess.run(['docker', 'ps', '--filter', 'ancestor=docker.elastic.co/elasticsearch/elasticsearch:8.10.1', '--format', '{{.Names}}'], capture_output=True, text=True)
    container_name = result.stdout.strip()
    # if container_name:
    #     copy_backup_to_container(backup_folder, container_name)
    # else:
    #     print("Aucun conteneur Elasticsearch en cours d'exécution trouvé.")
    #     return

    es_client = wait_for_elasticsearch(es_host, es_port, timeout=120)
    if es_client is None:
        print("Arrêt du script en raison de l'indisponibilité de Elasticsearch.")
        return

    # Récupérer la liste des index contenant 'name_exp' dans le nom
    indices = es_client.indices.get_alias(index=index_pattern)
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
                print(f"Erreur lors de la conversion de la date et l'heure pour l'index '{es_index}': {e}")
                continue
        else:
            print(f"Format du nom de l'index inattendu pour '{es_index}'.")
            continue

        duration_minutes = time_human_xp  # Durée en minutes du filtre

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

    # Créer les index dans la nouvelle instance Elasticsearch
    for es_index in index_names:
        # Charger le mapping sauvegardé
        with open(f'mappings/{es_index}_mapping.json', 'r', encoding='utf-8') as f:
            old_mapping = json.load(f)

        # Modify the settings to avoid replicas
        settings = {
            "settings": {
                "number_of_replicas": 0  # Set replicas to 0
            },
            "mappings": old_mapping['mappings']
        }

        # Créer l'index dans la nouvelle instance Elasticsearch
        if not new_es_client.indices.exists(index=es_index):
            new_es_client.indices.create(index=es_index, body=settings)
            print(f"Index '{es_index}' créé dans la nouvelle instance Elasticsearch avec 0 répliques.")

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


def objective(trial, name_xp : str):
    name_experiment = "msi_20-10-2024-12-02-28"
    time_human_xp = trial.suggest_categorical("time_human_xp", [0, 10, 20, 30, 40, 50])
    time_xp_auto = (60 - time_human_xp) * 60 # recuperate the time in seconds

    main('docker-compose-backup.yml', time_human_xp, name_experiment)

    with open(f'Optuna_results/{name_xp}.txt', 'a') as f:
        f.write(f"Time_human_xp: {time_human_xp}\n")

    performance = launch_run(default_llm_key="default_llm",
                             premium_llm_key="premium_llm",
                             problem_prompts_subdir="IR_CPS_TechSynthesis",
                             max_execution_time=time_xp_auto,
                             model_choice={"coach": "default_llm","coder": "premium_llm","critic": "default_llm","capitalizer": "default_llm"},
                             optuna_opti="coach",
                             name_exp=name_experiment,
                             arrayn_ret=True,
                             continue_each_loop=True)

    with open(f'Optuna_results/{name_xp}.txt', 'a') as f:
        f.write(f"performance: {max(performance)}\n")

    stop_containers('docker-compose-backup.yml')
    restart_original_container('elasticsearch/docker-compose.yml')
    wait_for_elasticsearch('localhost', 9200)

    return max(performance)





if __name__ == "__main__":
    global_main(objective, '', 'backup', initial_trials=[{'time_human_xp': 50}, {'time_human_xp': 40}, {'time_human_xp': 30}, {'time_human_xp': 20}, {'time_human_xp': 10}, {'time_human_xp': 0}], n_trials=5)
