import uuid
import json
import asyncio
import time
import os
import socket
from urllib.parse import urlparse, parse_qs
import threading
import shutil
from typing import Optional
import logging
import subprocess
from datetime import datetime
import websockets


class WebSocketServerConfig:
    """Configuration class for the WebSocket server."""
    def __init__(self, port: int = 6789, secret: Optional[str] = None, proxy_enabled: bool = False):
        """
        Initialize the WebSocket server configuration.

        Args:
            port (int): The port number for the WebSocket server. Defaults to 6789.
            secret (Optional[str]): A secret key for authentication.
                                    If True, a random secret is generated.
            proxy_enabled (bool): Whether to enable proxy functionality via l
                                    ocaltunnel. Defaults to False.
        """
        self.port: int = port
        self.proxy: Optional[bool] = proxy_enabled
        self.secret: Optional[str] = secret


class WebsocketServer:
    """WebSocket server class to handle client connections and messages."""

    def __init__(self, config: WebSocketServerConfig):
        """
        Initialize the WebSocket server.

        Args:
            config (WebSocketServerConfig): Configuration object for the server.
        """
        self.logger = logging.getLogger(__name__)
        # List of functions that can be executed in parallel
        self.parallel_functions = [
            "update_answer",
            "critic_answer",
            "get_tasks",
            "generate_best_improvement_suggestions",
            "goto_task"
        ]
        self.server_id = str(uuid.uuid4())
        # Stores agent monitors
        self.monitors = {}
        # Track current active monitor instances
        self.current_instances = {}
        # Track connected WebSocket clients
        self.connected_clients = set()
        # Count of messages sent
        self.message_count = 0

        # Get the hostname for logging
        self.host = socket.gethostname()
        # Port for the WebSocket server
        self.port = config.port

        # Secret for authentication
        self.secret = (
            None
            if not config.secret
            else (str(uuid.uuid4())[:28] if config.secret is True else config.secret)
        )
        # Whether proxy is enabled
        self.proxy_enabled = config.proxy
        # Proxy URL if localtunnel is used
        self.proxy_url = None
        # To store the localtunnel process
        self.process_lt = None
        self.max_connections = 10

        # Create logs directory if it doesn't exist
        if not os.path.exists("websocket_logs"):
            os.makedirs("websocket_logs")
        unique_id = os.environ.get('unique_id')
        if unique_id:
            self.log_filename = f'websocket_logs/websocketdata_{unique_id}_{self.port}.txt'
        else:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_filename = (
                f'websocket_logs/websocketdata_{self.host}_{self.port}_{date_str}.txt'
            )

        # Initialize the server thread
        # Event to stop the server
        self.stop_event = threading.Event()
        self.ws_thread = threading.Thread(target=self.run_server, daemon=True)
        self.ws_thread.start()

    def run_server(self):
        """Start the WebSocket server in a separate thread."""
        self.stop_event.clear()
        asyncio.run(self.main(self.stop_event))

    def stop_server(self):
        """Stop the WebSocket server and wait for the thread to terminate."""
        self.stop_event.set()
        self.ws_thread.join(timeout=5)
        self.ws_thread = None

    def add_monitor(self, monitor):
        """
        Add a monitor to the server.

        Args:
            monitor: The monitor object to add.
        """
        self.monitors[monitor.agent_name] = monitor
        self.current_instances[monitor.agent_name] = None

    def log_message(self, message: str, received: bool = True):
        """
        Log a message to the log file.

        Args:
            message (str): The message to log.
            received (bool): Whether the message was received (True) or sent (False).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{'IN' if received else 'OUT'} {timestamp} - {message}\n"
        with open(self.log_filename, "a",encoding="utf-8") as log_file:
            log_file.write(log_entry)

    async def handler(self, websocket, path):
        """
        Handle incoming WebSocket connections and messages.

        Args:
            websocket: The WebSocket connection object.
            path: The path of the WebSocket request.
        """
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

        # Reject connection if maximum connections are reached
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

                # Check if it is a function call
                if "function" in message_data:
                    agent_name = message_data.get("agent_name")
                    function_name = message_data.get("function")
                    params = message_data.get("params", {})
                    request_id = message_data.get("request_id")

                    if agent_name in self.monitors:
                        monitor = self.monitors[agent_name]
                        self.logger.info(
                            f"Executing function '{function_name}' for monitor '{agent_name}' "
                            f"with params: {params}"
                        )
                        if function_name in self.parallel_functions:
                            # Execute function asynchronously
                            asyncio.create_task(
                                self.execute_function_async(
                                    websocket, monitor, function_name, params, request_id
                                )
                            )
                            message = None  # Response will be sent asynchronously
                        else:
                            # Execute function synchronously
                            result = monitor.run_tool(function_name, params)
                            message = json.dumps({
                                "status": "success",
                                "message": None,
                                "result": result,
                                "function": function_name,
                                "request_id": request_id
                            })
                    else:
                        message = json.dumps({
                            "status": "error",
                            "message": f"Monitor '{agent_name}' not found",
                            "request_id": request_id
                        })

                # Broadcast the message to all connected clients except the sender
                for client in self.connected_clients:
                    if client != websocket and message is not None:
                        await client.send(message)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.logger.error(f"Error in WebSocket handler: {e}")
        finally:
            self.connected_clients.remove(websocket)

    async def execute_function_async(self, websocket, monitor, function_name, params, request_id):
        """
        Execute a function asynchronously and send the result back to the client.

        Args:
            websocket: The WebSocket connection object.
            monitor: The monitor object.
            function_name (str): The name of the function to execute.
            params (dict): The parameters for the function.
            request_id: The ID of the request.
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, monitor.run_tool, function_name, params
            )
            message = json.dumps({
                "status": "success",
                "message": None,
                "result": result,
                "function": function_name,
                "request_id": request_id
            })
            await websocket.send(message)
        except Exception as e:
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
                    self.logger.error(f"Port {self.port} is in use, trying the next one...")
                    self.port += 1

    def start_localtunnel(self, custom_host=None):
        """
        Start localtunnel if it's enabled and available on the system.

        Args:
            custom_host (Optional[str]): Custom host for localtunnel.
        """
        if self.proxy_enabled and shutil.which("lt") is not None:
            self.logger.info("Starting localtunnel...")
            try:
                lt_command = ["lt", "--port", str(self.port)]
                if custom_host:
                    lt_command.extend(["--host", custom_host])
                self.process_lt = subprocess.Popen(
                    lt_command,
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
                        self.logger.debug(f"Proxy established via localtunnel: {self.proxy_url}")
                        break
                    else:
                        self.logger.debug("Failed to retrieve the proxy URL from localtunnel.")
            except Exception as e:
                self.logger.error(f"Failed to start localtunnel: {e}")
        else:
            if self.proxy_enabled:
                self.logger.debug("Localtunnel is not installed. Please install it to enable proxy functionality.")
            else:
                self.logger.debug("Proxy functionality is disabled.")

    def display_urls(self):
        """Display the local and remote WebSocket URLs."""
        local_url = f"ws://localhost:{self.port}"
        if self.secret:
            local_url += f"?secret={self.secret}"
        self.logger.info(f"WebSocket Local URL: {local_url}")

        if self.proxy_url:
            remote_url = self.proxy_url
            if self.secret:
                remote_url += f"?secret={self.secret}"
            self.logger.info(
                f"WebSocket Remote URL via proxy: https://doxav.github.io/CollabFunctionsGPTCreator/IHMv5-Monaco.html?wsUrl={remote_url}"
            )

    async def main(self, stop_event):
        """
        Main coroutine to run the WebSocket server.

        Args:
            stop_event: Event to stop the server.
        """
        # Find available port before starting the server
        self.find_available_port()

        # Get the absolute path of IHMv5-Monaco.html
        current_directory = os.getcwd()
        hmi_file_path = os.path.join(current_directory, "Jquery_front", "IHMv5-Monaco.html")
        absolute_hmi_file_path = f"file://{hmi_file_path}"
        self.logger.info(f"Access to HMI via : {absolute_hmi_file_path}")

        # Start the WebSocket server
        server = await websockets.serve(self.handler, "127.0.0.1", self.port, ping_timeout=120)
        self.logger.info(f"WebSocket server started on port {self.port}")

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
        """
        Send a message to all connected clients.

        Args:
            message: The message to send.
        """
        no_client = True
        while no_client:
            if len(self.connected_clients) > 0:
                no_client = False
            else:
                self.display_urls()
                self.logger.info("Waiting for WebSocket client to connect")
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
                    self.logger.debug(f"Error sending message to client: {e}")

        try:
            loop = asyncio.get_running_loop()
            # If an event loop is running, schedule the coroutine
            asyncio.run_coroutine_threadsafe(send_to_clients(), loop)
        except RuntimeError:
            # No running event loop in this thread, so we can run the coroutine directly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send_to_clients())
        self.logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    def send_notasync_message(self, message):
        """
        Send a message without awaiting (fire and forget).

        Args:
            message: The message to send.
        """
        self.message_count += 1
        if isinstance(message, dict) and "sender_id" not in message:
            message['sender_id'] = self.server_id
        for client in self.connected_clients:
            asyncio.create_task(client.send(message))
