import json
import asyncio
import time

import websockets


class WebsocketServer:
    def __init__(self):
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
                # test if it is a function
                if "function" in message_data:
                    agent_name = message_data.get("agent_name")
                    function_name = message_data.get("function")
                    params = message_data.get("params", {})

                    if agent_name in self.monitors:
                        monitor = self.monitors[agent_name]
                        result = monitor.execute_function(function_name, params)
                        message = json.dumps({"status": "success", "result": result})
                        message = None
                    else:
                        message = json.dumps({"status": "error", "message": "Monitor not found"})
                        message = None
                for client in self.connected_clients:
                    if client != websocket and message is not None:
                        await client.send(message)
            return len(self.connected_clients)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)

    async def main(self, stop_event):
        server = await websockets.serve(self.handler, "localhost", 6789)
        while not stop_event.is_set():
            await asyncio.sleep(1)
        server.close()
        await server.wait_closed()

    def send_message(self, message):
        self.message_count += 1
        for client in self.connected_clients:
            asyncio.run(client.send(message))