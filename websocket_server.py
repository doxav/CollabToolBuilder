import json
import asyncio
import time
import os
import websockets
import subprocess
import uuid
from datetime import datetime
import socket

class WebsocketServer:
    def __init__(self):
        self.server_id = str(uuid.uuid4()) 
        self.monitors = {}  # Stores agent monitors
        self.current_instances = {}  # Track current active monitor instances
        self.connected_clients = set()
        self.message_count = 0
        self.host = socket.gethostname()  # Get the hostname for logging

    def add_monitor(self, monitor):
        self.monitors[monitor.agent_name] = monitor
        self.current_instances[monitor.agent_name] = None

    def log_message(self, message):
        # Create the log file name based on the host and current date
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"received_websocketdata_{self.host}_{date_str}.txt"
        
        # Append the timestamp and message to the log file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {message}\n"
        with open(log_filename, "a") as log_file:
            log_file.write(log_entry)

    async def handler(self, websocket, path):
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                self.log_message(message)  # Log every received message
                
                message_data = json.loads(message)
                if "sender_id" in message_data and message_data["sender_id"] == self.server_id:
                    return len(self.connected_clients)
                
                # Check if it is a function
                if "function" in message_data:
                    agent_name = message_data.get("agent_name")
                    function_name = message_data.get("function")
                    params = message_data.get("params", {})

                    if agent_name in self.monitors:
                        monitor = self.monitors[agent_name]
                        result = monitor.execute_function(function_name, params)
                        message = json.dumps({"status": "success", "message": None, "result": result, "function": function_name})
                    else:
                        message = json.dumps({"status": "error", "message": f"Monitor '{agent_name}' not found"})
                
                for client in self.connected_clients:
                    if client != websocket and message is not None:
                        await client.send(message)
            return len(self.connected_clients)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)

    async def main(self, stop_event):
        # Get the absolute path of IHMv4.html
        current_directory = os.getcwd()
        ihm_file_path = os.path.join(current_directory, "Jquery_front", "IHMv4.html")
        absolute_ihm_file_path = f"file://{ihm_file_path}"
        print(f"Access to IHM via : {absolute_ihm_file_path}")

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
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        clients = set(self.connected_clients)
        for client in clients:
            asyncio.run(client.send(message))
            print(message)

    def send_message(self, message):
        no_client = True
        while no_client:
            if len(self.connected_clients) > 0:
                no_client = False
            else:
                print("Waiting for WebSocket client to connect")
                time.sleep(1)
        self.message_count += 1
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        clients = set(self.connected_clients)

        async def send_to_clients():
            for client in clients:
                try:
                    await client.send(message)
                except Exception as e:
                    print(f"Error sending message to client: {e}")

        try:
            loop = asyncio.get_running_loop()
            # If an event loop is running, schedule the coroutine
            asyncio.run_coroutine_threadsafe(send_to_clients(), loop)
        except RuntimeError:
            # No running event loop in this thread, so we can run the coroutine directly
            asyncio.run(send_to_clients())
        print(message)
        
    def send_notasync_message(self, message):
        self.message_count += 1
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        for client in self.connected_clients:
            asyncio.create_task(client.send(message))
