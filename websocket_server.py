import json
import asyncio
import time
import os
import websockets
import subprocess

import uuid

class WebsocketServer:
    def __init__(self):
        self.server_id = str(uuid.uuid4()) 
        self.monitors = {}  # Stores agent monitors
        self.current_instances = {}  # Track current active monitor instances
        self.connected_clients = set()
        self.message_count = 0

    def add_monitor(self, monitor):
        self.monitors[monitor.agent_name] = monitor
        self.current_instances[monitor.agent_name] = None

    async def handler(self, websocket, path):
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                message_data = json.loads(message)
                if "sender_id" in message_data and message_data["sender_id"] == self.server_id:
                    return len(self.connected_clients)
                # test if it is a function
                if "function" in message_data:
                    agent_name = message_data.get("agent_name")
                    function_name = message_data.get("function")
                    params = message_data.get("params", {})

                    if agent_name in self.monitors:
                        monitor = self.monitors[agent_name]
                        result = monitor.execute_function(function_name, params)
                        message = json.dumps({"status": "success", "result": result, "function": function_name})
                        #message = None
                    else:
                        message = json.dumps({"status": "error", "message": "Monitor not found"})
                        #message = None
                for client in self.connected_clients:
                    if client != websocket and message is not None:
                        await client.send(message)
            return len(self.connected_clients)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)

    async def main(self, stop_event):
        # Obtenir le chemin absolu du fichier IHMv4.html
        current_directory = os.getcwd()
        ihm_file_path = os.path.join(current_directory, "Jquery_front", "IHMv4.html")
        absolute_ihm_file_path = f"file://{ihm_file_path}"
        # Afficher le lien dans le terminal
        print(f"Access to IHM via : {absolute_ihm_file_path}")
        try:
            windows_ihm_file_path = subprocess.check_output(["wslpath", "-w", ihm_file_path]).decode("utf-8").strip()
            absolute_ihm_file_path = "file:///" + windows_ihm_file_path.replace('\\', '/')
            print(f"or on Windows WSL via : {absolute_ihm_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Windows WSL path not found : {e}")

        server = await websockets.serve(self.handler, "localhost", 6789)
        while not stop_event.is_set():
            await asyncio.sleep(1)
        server.close()
        await server.wait_closed()

    def send_message(self, message):
        no_client = True
        while no_client:
            if len(self.connected_clients) > 0:
                no_client = False
            else:
                print("Waiting for WebSocket client to connect")
                time.sleep(1)
        self.message_count += 1
        # test if message is dict
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        for client in self.connected_clients:
            asyncio.run(client.send(message))

    def send_notasync_message(self, message):
        self.message_count += 1
        # test if message is dict
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        for client in self.connected_clients:
            asyncio.create_task(client.send(message))
