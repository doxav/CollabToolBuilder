import os
import sys
import subprocess
import time
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Exécute un programme dans des conteneurs Docker temporaires.')
    parser.add_argument('--files', nargs='+', help='Liste des fichiers à traiter', required=False)
    parser.add_argument('--folder', help='Dossier contenant les fichiers à traiter', required=False)
    parser.add_argument('--docker_compose_file', help='Chemin vers docker-compose.yml', default='docker-compose.yml')
    args = parser.parse_args()
    return args

def stop_existing_containers():
    print("Vérification de l'existence de conteneurs Elasticsearch/Kibana en cours d'exécution...")
    # Récupérer les IDs des conteneurs Elasticsearch
    result = subprocess.run(['docker', 'ps', '-q', '--filter', 'name=elasticsearch'], capture_output=True, text=True)
    elasticsearch_containers = result.stdout.strip().split('\n') if result.stdout.strip() else []

    # Récupérer les IDs des conteneurs Kibana
    result = subprocess.run(['docker', 'ps', '-q', '--filter', 'name=kibana'], capture_output=True, text=True)
    kibana_containers = result.stdout.strip().split('\n') if result.stdout.strip() else []

    containers_to_stop = elasticsearch_containers + kibana_containers

    if containers_to_stop:
        print("Arrêt des conteneurs Elasticsearch/Kibana existants sans les supprimer...")
        subprocess.run(['docker', 'stop'] + containers_to_stop)
        print("Conteneurs Elasticsearch/Kibana existants arrêtés.")
    else:
        print("Aucun conteneur Elasticsearch/Kibana en cours d'exécution.")

def start_containers(docker_compose_file):
    # Arrêter les conteneurs existants s'il y en a
    stop_existing_containers()

    print("Démarrage des conteneurs Docker...")
    subprocess.check_call(['docker', 'compose', '-f', docker_compose_file, 'up', '-d'])
    # Attendre que Elasticsearch soit prêt
    print("Attente du démarrage de Elasticsearch...")
    time.sleep(30)  # Ajustez le temps d'attente si nécessaire
    print("Conteneurs démarrés.")

def stop_containers(docker_compose_file):
    print("Arrêt des conteneurs Docker...")
    subprocess.check_call(['docker', 'compose', '-f', docker_compose_file, 'down'])
    print("Conteneurs arrêtés et supprimés.")

def new_docker():
    args = parse_args()

    # Collecte des fichiers
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"Le dossier {args.folder} n'existe pas.")
            sys.exit(1)
        files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]
    elif args.files:
        files = args.files
        for f in files:
            if not os.path.isfile(f):
                print(f"Le fichier {f} n'existe pas.")
                sys.exit(1)
    else:
        print("Vous devez fournir soit --files soit --folder.")
        sys.exit(1)

    # Création d'un dossier temporaire pour stocker les fichiers
    temp_dir = 'temp_files'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    for f in files:
        shutil.copy(f, temp_dir)

    try:
        # Démarrage des conteneurs Docker
        start_containers(args.docker_compose_file)

    except subprocess.CalledProcessError as e:
        print(f"Une erreur s'est produite : {e}")


if __name__ == '__main__':
    new_docker()
