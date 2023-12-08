from elasticsearch import Elasticsearch, helpers
import time
from datetime import datetime
from utils.llm_utils import UnifiedVectorDB
from config import *
import socket

elastic_connection = Elasticsearch(elastic_url_port)

def view_multiple_generic_geo(docs_json):
    """
    Inserts multiple geospatial data points into the 'generic-geo-index' and returns a link to view the data in Kibana.

    :param docs_json: A JSON string representing an array of documents. Each document should contain 'location' (with 'lat' and 'lon'), 'title', 'timestamp', and 'description' fields. Example of docs_json: '[{"location": {"lat": 48.8566, "lon": 2.3522}, "title": "Paris, France", "timestamp": "2023-11-10T15:00:00", "description": "Event in Paris."}]'
    :type docs_json: str
    :return: A link to view the data in Kibana.
    :rtype: str
    """
    import json
    index = "generic-geo-index"
    
    # Définir le mapping
    mapping = {
        "mappings": {
            "properties": {
                "location": {"type": "geo_point"},
                "title": {"type": "text"},
                "timestamp": {"type": "date"},
                "description": {"type": "text"},
                "host": {"type":"keyword"},
                "creation_ts": {"type": "date"}
            }
        }
    }

    # Créer l'index avec le mapping
    if elastic_connection.indices.exists(index=index):
        elastic_connection.delete_by_query(index=index, body={"query": {"match_all": {}}})
    else:
        elastic_connection.indices.create(index=index, body=mapping, ignore=400)  # ignore 400 signifie ignorer l'erreur si l'index existe déjà

    time.sleep(0.1)

    docs = json.loads(docs_json) if isinstance(docs_json, str) else docs_json
    for doc in docs:
        doc["creation_ts"] = datetime.now().isoformat()
        doc["host"] = socket.gethostbyname(socket.gethostname())

    actions = [
        {
            "_index": index,
            "_source": doc
        }
        for doc in docs
    ]
    helpers.bulk(elastic_connection, actions)
    return f"View data at {kibana_url_port}/app/maps/map/4b531030-82f4-11ee-9ef2-a71556ea9523"


def view_multiple_generic_table(docs_json):
    """
    Inserts multiple documents into the 'generic-table-index' and returns a link to view the data in Kibana.

    :param docs_json: A JSON string representing an array of documents. Each document should contain 'timestamp', 'category', 'title', 'description', and 'id' fields. Example of docs_json: '[{"timestamp": "2023-11-13T12:00:00", "category": "Technology", "title": "Innovations in AI", "description": "Exploring AI trends.", "id": "doc1"}]'
    :type docs_json: str
    :return: A link to view the data in Kibana.
    :rtype: str
    """
    import json
    index = "generic-table-index"

    mapping = {
        "mappings": {
            "properties": {
                "timestamp": { "type": "date" },
                "category": { "type": "keyword" },
                "title": { "type": "text" },
                "description": { "type": "text" },
                "id": { "type": "keyword" },
                "host": {"type":"keyword"},
                "creation_ts": {"type": "date"}
            }
        }
    }

    # Créer l'index avec le mapping
    if elastic_connection.indices.exists(index=index):
        elastic_connection.delete_by_query(index=index, body={"query": {"match_all": {}}})
    else:
        elastic_connection.indices.create(index=index, body=mapping, ignore=400)  # ignore 400 signifie ignorer l'erreur si l'index existe déjà

    time.sleep(0.1)

    docs = json.loads(docs_json) if isinstance(docs_json, str) else docs_json
    for doc in docs:
        doc["creation_ts"] = datetime.now().isoformat()
        doc["host"] = socket.gethostbyname(socket.gethostname())

    actions = [
        {
            "_index": index,
            "_source": doc
        }
        for doc in docs
    ]
    helpers.bulk(elastic_connection, actions)
    
    return f"View data at {kibana_url_port}/app/lens#/edit/b3651fb0-8308-11ee-9ef2-a71556ea9523?_g=(filters:!(),refreshInterval:(pause:!t,value:60000),time:(from:'1950-04-24T01:00:00.000Z',to:'2050-04-24T00:00:00.000Z'))"



# {
#   "tool": "ElasticsearchDataHandler",
#   "functions": [
#     {
#       "name": "view_multiple_generic_geo",
#       "description": "Inserts multiple geospatial data points into the 'generic-geo-index' and returns a link to view the data in Kibana.",
#       "parameters": [
#         {
#           "name": "docs_json",
#           "type": "string",
#           "description": "A JSON string representing an array of documents. Each document should contain 'location' (with 'lat' and 'lon'), 'title', 'timestamp', and 'description' fields."
#         }
#       ],
#       "returns": {
#         "type": "string",
#         "description": "Link to view the data in Kibana."
#       }
#     },
#     {
#       "name": "view_multiple_generic_table",
#       "description": "Inserts multiple documents into the 'generic-table-index' and returns a link to view the data in Kibana.",
#       "parameters": [
#         {
#           "name": "docs_json",
#           "type": "string",
#           "description": "A JSON string representing an array of documents. Each document should contain 'timestamp', 'category', 'title', 'description', and 'id' fields."
#         }
#       ],
#       "returns": {
#         "type": "string",
#         "description": "Link to view the data in Kibana."
#       }
#     }
#   ]
# }
