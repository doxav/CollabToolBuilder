#                           Création de la pipeline Open WebUI 

## Liste des commandes

###  - Réaliser les installations de paquets 

    pip install neo4j scikit-learn openai llama-index llama-index
    -llms-ollama llama-index-embeddings-ollama

###  - Lancer l'image Neo4J avec l'accès à la base graph

![graph neo4j](graph_readme.png)

###  - Ajouter la clée d'accès OpenAI
    OPENAI_API_KEY=your-openai-api-key

###  - Démarrer Ollama 

    ollama serve --model llama3_8b

###  - Configurer les variables d'environnement 

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=password
        
    LLAMAINDEX_OLLAMA_BASE_URL=http://localhost:11434
    LLAMAINDEX_MODEL_NAME=llama3_8b
    LLAMAINDEX_EMBEDDING_MODEL_NAME=nomic-embed-text

### Lancer OpenWebUI via Docker 

    docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

    lancer bash ./start.sh from pipelines issue de ::
            git clone https://github.com/open-webui/pipelines.git

### Paramétrage sous OpenWebUi

    ouvrir localhost:8080, dans Admin Panel, établir les connections nécessaires: 

            http://localhost:11434 pour ollama

