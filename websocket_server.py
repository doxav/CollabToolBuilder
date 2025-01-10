import json
import asyncio
import time
import os
import websockets
import subprocess
import uuid
from datetime import datetime
import socket
from urllib.parse import urlparse, parse_qs
import threading
import shutil  # To check if localtunnel is available

class WebsocketServer:
    parallel_functions = ["updateAnswer", "criticAnswer", "get_tasks", "generate_best_improvement_suggestions"]
    def __init__(self, port=6789, secret=None, proxy_enabled=False, unique_id=None):
        self.server_id = str(uuid.uuid4())
        self.monitors = {}  # Stores agent monitors
        self.current_instances = {}  # Track current active monitor instances
        self.connected_clients = set()
        self.message_count = 0
        self.host = socket.gethostname()  # Get the hostname for logging
        self.port = port
        # if secret is None or False set None, if True generate a random secret of 17 characters, else use the provided secret
        self.secret = None if not secret else (str(uuid.uuid4())[:28] if secret is True else secret)
        self.proxy_enabled = proxy_enabled
        self.proxy_url = None
#        self.loop = asyncio.new_event_loop()
        self.process_lt = None  # To store the localtunnel process
        self.max_connections = 10
        if unique_id:
            self.log_filename = f'websocketdata_{unique_id}_{self.port}.txt'
        else:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_filename = f'websocketdata_{self.host}_{self.port}_{date_str}.txt'

    def add_monitor(self, monitor):
        self.monitors[monitor.agent_name] = monitor
        self.current_instances[monitor.agent_name] = None

    def log_message(self, message, received=True):
        # Append the timestamp and message to the log file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{'IN' if received else 'OUT'} {timestamp} - {message}\n"
        with open(self.log_filename, "a") as log_file:
            log_file.write(log_entry)

    async def handler(self, websocket, path):
        # Parse query parameters
        query_params = parse_qs(urlparse(path).query)
        if self.secret:
            received_secret = query_params.get("secret", [None])[0]
            if received_secret != self.secret:
                try:
                    await websocket.close()
                except Exception:
                    pass  # Ignore any exceptions during closing
                return

        if len(self.connected_clients) >= self.max_connections:
            try:
                await websocket.close()
            except Exception:
                pass
            return
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                message_data = json.loads(message)
                if "sender_id" in message_data and message_data["sender_id"] == self.server_id:
                    continue  # Ignore messages sent by the server itself

                self.log_message(message, received=True)  # Log every received message

                # Check if it is a function
                if "function" in message_data:
                    agent_name = message_data.get("agent_name")
                    function_name = message_data.get("function")
                    params = message_data.get("params", {})
                    request_id = message_data.get("request_id")  # Add this line

                    if agent_name in self.monitors:
                        monitor = self.monitors[agent_name]
                        print(
                            f"Executing function '{function_name}' for monitor '{agent_name}' with params: {params}")
                        if function_name in self.parallel_functions:  # Add this condition
                            # Execute function asynchronously
                            asyncio.create_task(
                                self.execute_function_async(websocket, monitor, function_name, params,
                                                            request_id))  # Add this line
                            message = None  # Since response will be sent asynchronously
                        else:
                            # Execute function synchronously
                            result = monitor.execute_function(function_name, params)
                            message = json.dumps({
                                "status": "success",
                                "message": None,
                                "result": result,
                                "function": function_name,
                                "request_id": request_id  # Add this line
                            })
                    else:
                        message = json.dumps({
                            "status": "error",
                            "message": f"Monitor '{agent_name}' not found",
                            "request_id": request_id  # Add this line
                        })

                for client in self.connected_clients:
                    if client != websocket and message is not None:
                        await client.send(message)
        except Exception as e:
            # Print exception details
            print(f"Error in WebSocket handler: {e}")
            pass
        finally:
            self.connected_clients.remove(websocket)

    async def execute_function_async(self, websocket, monitor, function_name, params, request_id):
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, monitor.execute_function, function_name, params)
            message = json.dumps({
                "status": "success",
                "message": None,
                "result": result,
                "function": function_name,
                "request_id": request_id
            })
            await websocket.send(message)
        except Exception as e:
            # Handle exceptions and send error message
            error_message = json.dumps({
                "status": "error",
                "message": f"Error executing function '{function_name}': {e}",
                "function": function_name,
                "request_id": request_id
            })
            await websocket.send(error_message)

    def find_available_port(self):
        """Find the next available port starting from the current self.port."""
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                try:
                    sock.bind(("localhost", self.port))
                    sock.close()
                    return  # Port is available
                except OSError:
                    print(f"Port {self.port} is in use, trying the next one...")
                    self.port += 1

    def start_localtunnel(self):
        """Start localtunnel if it's enabled and available on the system."""
        if self.proxy_enabled and shutil.which("lt") is not None:
            print("Starting localtunnel...")
            try:
                self.process_lt = subprocess.Popen(
                    ["lt", "--port", str(self.port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # Read the remote URL from localtunnel output
                while True:
                    line = self.process_lt.stdout.readline()
                    if not line:
                        break
                    if "your url is:" in line:
                        self.proxy_url = line.strip().split("your url is:")[1].strip()
                        print(f"Proxy established via localtunnel: {self.proxy_url}")
                        break
                    else:
                        print("Failed to retrieve the proxy URL from localtunnel.")
            except Exception as e:
                print(f"Failed to start localtunnel: {e}")
        else:
            if self.proxy_enabled:
                print("Localtunnel is not installed. Please install it to enable proxy functionality.")
            else:
                print("Proxy functionality is disabled.")

    def display_urls(self):
        """Display the local and remote WebSocket URLs."""
        local_url = f"ws://localhost:{self.port}"
        if self.secret:
            local_url += f"?secret={self.secret}"
        print(f"WebSocket Local URL: {local_url}")

        if self.proxy_url:
            remote_url = self.proxy_url
            if self.secret:
                remote_url += f"?secret={self.secret}"
            print(f"WebSocket Remote URL via proxy: https://doxav.github.io/CollabFunctionsGPTCreator/IHMv5-Monaco.html?wsUrl={remote_url}")

    async def main(self, stop_event):
        # Find available port before starting the server
        self.find_available_port()

        # Get the absolute path of IHMv5-Monaco..html
        current_directory = os.getcwd()
        hmi_file_path = os.path.join(current_directory, "Jquery_front", "IHMv5-Monaco..html")
        absolute_hmi_file_path = f"file://{hmi_file_path}"
        print(f"Access to HMI via : {absolute_hmi_file_path}")

        # Start the WebSocket server
        server = await websockets.serve(self.handler, "127.0.0.1", self.port, ping_timeout=120)
        print(f"WebSocket server started on port {self.port}")

        # Start localtunnel if proxy is enabled
        self.start_localtunnel()

        # Display the WebSocket URLs
        self.display_urls()

        # Keep the server running until stop_event is set
        while not stop_event.is_set():
            await asyncio.sleep(1)
        server.close()
        await server.wait_closed()

    def send_message(self, message):
        """Send a message to all connected clients."""
        no_client = True
        while no_client:
            if len(self.connected_clients) > 0:
                no_client = False
            else:
                self.display_urls()
                print("Waiting for WebSocket client to connect")
                time.sleep(2)
        self.message_count += 1
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        clients = set(self.connected_clients)

        async def send_to_clients():
            for client in clients:
                try:
                    await client.send(message)
                    self.log_message(message, received=False)
                except Exception as e:
                    print(f"Error sending message to client: {e}")

        try:
            loop = asyncio.get_running_loop()
            # If an event loop is running, schedule the coroutine
            asyncio.run_coroutine_threadsafe(send_to_clients(), loop)
        except RuntimeError:
            # No running event loop in this thread, so we can run the coroutine directly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send_to_clients())
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    def send_notasync_message(self, message):
        """Send a message without awaiting (fire and forget)."""
        self.message_count += 1
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        for client in self.connected_clients:
            asyncio.create_task(client.send(message))
