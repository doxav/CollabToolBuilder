import asyncio
import websockets
from flask import Flask, jsonify, request
from flask_cors import CORS
from threading import Thread

from config import *
from utils.llm_utils import HumanLLMMonitor, UnifiedVectorDB

UnifiedVectorDB.es_url = elastic_url_port
UnifiedVectorDB.es_user = elastic_user
UnifiedVectorDB.es_password = elastic_password

app = Flask(__name__)
"""
The error you're encountering is due to the same-origin policy, which prevents JavaScript code running on one 
origin (domain) from accessing resources from a different origin. In your case, http://localhost:63342 is trying to 
access http://127.0.0.1:5000, and this is being blocked by the browser's CORS policy.

To resolve this issue, you need to enable CORS on your server. Here's how you can do it if you're using Flask for 
your API:
"""
CORS(app)

connected_clients = set()


def run_flask():
    app.run(debug=True, use_reloader=False, port=5000)


@app.route("/api/hello", methods=["GET"])
def hello():
    return "Hello, World!"


@app.route("/api/history", methods=["GET"])
def history():
    function_name = request.args.get('function_name', None)
    agent_name = request.args.get('agent_name', None)

    # Filtrer les résultats en fonction des paramètres
    results = HumanLLMMonitor.getPreviousResults(function_name=function_name, agent_name=agent_name)

    return jsonify(results)


@app.route("/api/criticanswer", methods=["POST"])
def criticanswer():
    data = request.get_json()
    function_name = data['function_name']
    agent_name = data['agent_name']
    question = data['question']
    answer = data['answer']
    HumanLLMMonitor.criticAnswer(function_name, agent_name, question, answer)
    return jsonify({"status": "success"})

@app.route("/api/changeDefaultLLM", methods=["POST"])
def changeDefaultLLM():
    data = request.get_json()
    default_llm = data['default_llm']
    HumanLLMMonitor.changeDefaultLLM(default_llm)
    return jsonify({"status": "success"})


async def handler(websocket, path):
    # Register the new client
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            # Broadcast the message to all connected clients
            for client in connected_clients:
                if client != websocket:
                    await client.send(message)
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)


async def main():
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    async with websockets.serve(handler, "localhost", 6789):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
