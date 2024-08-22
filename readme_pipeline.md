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

    => sans ollama : docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

      ===> http://localhost:3000


    => avec ollama : docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main

      ===> http://localhost:8080        



    lancer bash ./start.sh from pipelines issue de ::
            git clone https://github.com/open-webui/pipelines.git


### Paramétrage sous OpenWebUi


    avec ollama : the WebUI docker container not being able to reach the Ollama server at 127.0.0.1:11434 (host.docker.internal:11434) inside the container . Use the --network=host flag in your docker command to resolve this. Note that the port changes from 3000 to 8080, resulting in the link: .

    ouvrir localhost:3000, se connecter, puis dans Admin Panel, établir les connections nécessaires: 

            https://api.openai.com/v1   et mettre sa clée openai

            http://localhost:9100   pwd : 0p3n-w3bu!

            ou via docker : http://host.docker.internal:9100   pwd : 0p3n-w3bu!

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


