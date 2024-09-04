# Collab HumanLLM functions creation XP 2024

## Prerequisites 

- Python 3.10+
- Venv or Conda env
- VSCode

## Installation

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

MODIFY CONFIG.PY with the IP that will be given to be able to access the local ElasticSearch data server

## Launching the learning to develop new functions

To start the collaborative development process on local IHM for new commands, execute the command:

```
python learn.py
python websocket_server.py
```
You can start creating functions or pipelines by using IHMv1.html or IHMv2.html


## The basic architecture

*1. Database (documentary and vectorial)* ElasticSearch: capitalizes our trials, stores new functions or pipelines (can perform similarity search), is used with Kibana to temporarily store conversational agent results and visualize them

*2. Capacity/Functions Development Agent System* Learn.py: iterative process of proposing new tasks to automate, coding, validation (go or new coding attempt), capitalization (indexing for reuse).
![Learning Loop](readme_learningloop.gif)

*PRINCIPLES*
1. Proposal for new task => adapt the proposed task (short term) or adapt the agent's role (long term)
2. Code proposal => adapt the code (short term) or adapt the agent's role (long term)
3. Critique proposal => modify the critique (short term) or correct the agent's role (long term)
4. Proposal for task indexing description => correct the description (short term) or correct the agent's role (long term)

*Human LLM Mechanism*: (TO EDIT CODE OR TEXT: a file is automatically opened in VSCode, it is by closing it that the text is validated and the process continues) \
1. control what will be sent to the agent Before () \
> A. Modify agent's system prompt (role & global context, constraints, examples). \
> B. Add instruction or information to agent. \
> C. Skip and set LLM output from recent outputs or manually define it. \
> D. Log comments. \
> E. See all previous results for this agent. \
> F. See previous MODIFIED/SCORED/COMMENTED results for this agent. \
> G. Skip human actions for N rounds. \
> H. Exit program. \
> P. Proceed to inference using a PREMIUM LLM. \

2. control the agent's "After" output to modify it or learn to improve what is sent to it () \
> A. Manually set the answer/OUTPUT (I don't want to try to improve agent's system prompt). \
> B. Criticize this answer to get an improved answer. \
> C. Find a better Prompt by providing critic and/or ideal answer. \
> D. ANNOTATE: evaluate & comment answer (Score between 0(worst)-1(top), and explain) to improve future results by using scored/commented examples. \
> E. Go back BEFORE inference to improve system prompt or add information to user message. \
> G. Skip human actions for N rounds. \
> H. Exit program. \

The validated functions that we will capitalize are placed in the **functions directory** or **pipelines/pipelines**
Prompts contain the "system prompts" sent to agents but it is much preferable to modify them from learn.py (via the Human LLM mechanism). \


# Pipeline integration for Anomalies Solver

###  Launch Neo4J database load on your machine 
![graph neo4j](graph_readme.png)

### Launch OpenWebUI with Docker 

    => without ollama 
    
    docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

        ===> watch on  http://localhost:3000


    => avec ollama : 

    docker run -d   --network=host   -v open-webui:/app/backend/data   --add-host=host.docker.internal:host-gateway   -e PIPELINES_URLS="$(for file in /pipelines/pipelines/*; do echo -n "$file,"; done | sed 's/,$//')"   -e OLLAMA_BASE_URL=http://127.0.0.1:11434   -v /path/to/pipelines:/app/pipelines   --name pipelines-combin   --restart always   ghcr.io/open-webui/pipelines:main

        ===> watch on http://localhost:8080        


    

#### Mais il est plus simple de lance via $run_docker.sh (lance le docker openwebui, le script start.sh qui charge les pipelines, les serveurs uvicorns serveillent directement les modifiactions apportées à /pipelines/pipelines)


![maj pipe](v2.png)

```
git clone https://github.com/open-webui/pipelines.git
bash run_docker.sh
bash ./start.sh

```

### OpenWebUi settings


    with ollama : the WebUI docker container may not being able to reach the Ollama server at 127.0.0.1:11434 (host.docker.internal:11434) inside the container . Use the --network=host flag in your docker command to resolve this. Note that the port changes from 3000 to 8080, resulting in the link.

    open localhost:8080, register or login, and then in Admin Panel, set the following mandatory connections: 

            https://api.openai.com/v1   (add your personnal openai key)

            http://localhost:9100   pwd : 0p3n-w3bu!

                    NB: if you use docker : http://host.docker.internal:9100   pwd : 0p3n-w3bu!

            http://localhost:11434 for ollama

    
        depending on the pipelines valves, you may have to fill the missing connections informations, such as : 
                Llamaindex Ollama Base Url              http://localhost:11434
                Llamaindex Model Name                   llama3_8b
                Llamaindex Embedding Model Name         nomic-embed-text
                Neo4J Uri                               bolt://localhost:7687
                Neo4J User                              neo4j
                Neo4J Password                          password
                Openai Api Key                          your-key-api


