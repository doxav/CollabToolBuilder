
# MANUAL - Real-Time Communication Interface

## Table of Contents
1. [Overview](#overview)
2. [WebSocket Setup and Communication](#websocket-setup-and-communication)
3. [Interface Layout](#interface-layout)
4. [Message Handling](#message-handling)
5. [Dynamic Content Features](#dynamic-content-features)
6. [User Interface Interactions](#user-interface-interactions)
7. [Custom Evaluations](#custom-evaluations)
8. [Settings Application](#settings-application)
9. [Error Handling and Recovery](#error-handling-and-recovery)

---

## 1. Overview
This document describes the Real-Time Communication Interface implemented in `IHMv3.html`. The interface facilitates real-time communication between a front-end user interface and backend systems via WebSocket. It supports multiple agents, message handling, evaluations, and dynamic user interface updates.

## 2. WebSocket Setup and Communication
The interface connects to a WebSocket server at `ws://localhost:6789`. This connection allows real-time communication for sending and receiving messages between the front end and agents.

### WebSocket Connection:
- **Initialization**:
    ```javascript
    let socket = new WebSocket('ws://localhost:6789');
    socket.onopen = function() {
        console.log('WebSocket connection established');
    };
    ```

- **Sending Messages**:
    Messages are sent in JSON format with an `agent_name`, `function`, and `params`.
    ```javascript
    let message = {
        agent_name: "AgentName",
        function: "FunctionName",
        params: {param1: "value"}
    };
    socket.send(JSON.stringify(message));
    ```

- **Receiving Messages**:
    Messages from the WebSocket are received and processed based on their `message_type`.
    ```javascript
    socket.onmessage = function(event) {
        let data = JSON.parse(event.data);
        handleMessage(data);
    };
    ```

## 3. Interface Layout
The interface is divided into three main sections:
- **History Sidebar**: Displays historical information.
- **Main Content**: Contains the message accordion, user input field, and filter options.
- **Settings Sidebar**: Provides options to configure the behavior of agents, including selecting default and premium LLMs and setting temperature.

The layout is implemented using a Flexbox design to ensure responsiveness. Hidden sidebars are toggled as needed.

## 4. Message Handling
The interface handles various types of messages:
- **Inference Results**: Displays multiple solutions in columns. Solutions include buttons for expansion, evaluation, and pagination when there are more than four solutions.
- **Evaluation Results**: Supports agents that return scores or compliance results. Custom logic is applied depending on the agent's evaluation type.
- **Prompt Inputs**: The interface dynamically generates input fields (e.g., number of rounds, parallel inferences).

## 5. Dynamic Content Features
### Accordion:
The main content is displayed in an accordion, which updates automatically when new messages are received. It supports filtering by agent or message type.

### Pagination:
If multiple solutions are presented, a pagination system is used to display up to four solutions per page.

### Expand/Collapse:
Solutions can be expanded for a more detailed view using the expand button, and collapsed back as needed.

## 6. User Interface Interactions
- **Filter Bars**: Two filter bars (top and bottom) allow the user to filter messages by agent and message type. These filter bars are synchronized.
- **Settings Sidebar**: Contains controls for changing the default LLM, premium LLM, and adjusting the inference temperature.
- **Auto-Focus**: A checkbox enables auto-focus on new messages, automatically scrolling to the latest update.
- **Annotation Tool**: Users can highlight and annotate parts of the message for review or action.

## 7. Custom Evaluations
The interface supports custom evaluations for different agents:
- **Score-based Evaluations**: Solutions can be scored using a slider, and the score is sent back to the server.
- **Compliance-based Evaluations**: Solutions are evaluated for compliance using a dropdown menu.

After evaluations, additional options for deleting or regenerating solutions are provided.

## 8. Settings Application
The settings sidebar allows users to modify configurations such as:
- **Default and Premium LLMs**: Users can select which language model or chain of models to use for inference.
- **Temperature**: The temperature slider adjusts the randomness of the model's responses. Once settings are applied, they are sent to the server via WebSocket.

Example:
```javascript
let settings = {
    defaultLLM: $("#default-llm").val(),
    premiumLLM: $("#premium-llm").val(),
    temperature: $("#temperature").val()
};
```

## 9. Error Handling and Recovery
The system is designed to handle WebSocket disconnections gracefully. If the connection is lost, the system attempts to reconnect every 2 seconds:
```javascript
socket.onclose = function() {
    console.log('WebSocket connection closed');
    setTimeout(function() {
        socket = new WebSocket('ws://localhost:6789');
    }, 2000);
};
```

Errors during message parsing or sending are logged to the console for troubleshooting.