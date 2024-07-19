import asyncio
import websockets
import json
import random


async def test_server(websocket, path):
    try:
        while True:
            # Simulate different types of messages
            message_types = ["info", "warning", "error", "input", "menu", "column"]
            agent_names = ["Agent1", "Agent2", "Agent3"]

            message_type = random.choice(message_types)
            agent_name = random.choice(agent_names)

            if message_type == "column":
                message = {
                    "message": f"This is a {message_type} message from {agent_name}",
                    "agent_name": agent_name,
                    "message_type": message_type,
                    "column_id": f"col_{random.randint(1, 5)}",
                    "column_max": 5,
                    "input": False,
                    "user_id": "test_user"
                }
            elif message_type == "menu":
                message = {
                    "message": "BEFORE\n[A] Option A\n[B] Option B\n[C] Option C",
                    "agent_name": agent_name,
                    "message_type": message_type,
                    "input": True,
                    "user_id": "test_user"
                }
            elif message_type == "input":
                message = {
                    "message": "Please provide input:",
                    "agent_name": agent_name,
                    "message_type": message_type,
                    "input": True,
                    "user_id": "test_user"
                }
            else:
                message = {
                    "message": f"This is a {message_type} message from {agent_name}",
                    "agent_name": agent_name,
                    "message_type": message_type,
                    "input": False,
                    "user_id": "test_user"
                }

            await websocket.send(json.dumps(message))
            await asyncio.sleep(5)  # Wait for 5 seconds before sending the next message
    except websockets.exceptions.ConnectionClosed:
        pass


start_server = websockets.serve(test_server, "localhost", 6789)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()