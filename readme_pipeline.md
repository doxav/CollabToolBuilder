#                           Création de la pipeline Open WebUI 

## Liste des commandes

###  - Réaliser les installations de paquets 

    pip install neo4j scikit-learn openai llama-index llama-index
    -llms-ollama llama-index-embeddings-ollama

###  - Lancer l'image Neo4J avec l'accès à la base graph

![graph neo4j](graph_readme.png)

###  - Ajouter la clée d'accès OpenAI
    OPENAI_API_KEY=your-openai-api-key

### Lancer OpenWebUI via Docker 

    docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

    lancer bash ./start.sh from pipelines issue de ::
            git clone https://github.com/open-webui/pipelines.git

### Paramétrage sous OpenWebUi

    ouvrir localhost:3000, se connecter, puis dans Admin Panel, établir les connections nécessaires: 

            https://api.openai.com/v1   et mettre sa clée openai
            
            http://host.docker.internal:9100   pwd : 0p3n-w3bu!

            NB : If your Open WebUI is running in a Docker container, replace localhost with host.docker.internal in the API URL

            http://localhost:11434 pour ollama

    Dans "pipelines", charger la pipelines.py à partir de github
         le github doit contenir .env avec la clée openai,
         et un .gitignore pour eviter de la publier

        exemple de github : https://github.com/doxav/CollabFunctionsGPTCreator/blob/Jira-FA/anomaly_retrieval_pipeline.py

        et régler les différents paramètres : 
                Llamaindex Ollama Base Url              http://localhost:11434
                Llamaindex Model Name                   llama3_8b
                Llamaindex Embedding Model Name         nomic-embed-text
                Neo4J Uri                               bolt://localhost:7687
                Neo4J User                              neo4j
                Neo4J Password                          password
                Openai Api Key                          your-key-api


