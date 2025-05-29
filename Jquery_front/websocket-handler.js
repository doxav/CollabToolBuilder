class WebSocketHandler {
  constructor(url) {
    this.url = url;
    this.socket = null;

    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;

    this.initWebSocket();
  }

  initWebSocket() {
    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      console.log("Connected to WebSocket");
      this.reconnectAttempts = 0; // Reset reconnect count
    };

    this.socket.onmessage = (event) => this.handleMessage(event);
    this.socket.onerror = (error) => console.error("WebSocket error:", error);
    this.socket.onclose = () => this.reconnect();
  }

  handleMessage(event) {
    try {
      if (!event.data) {
        console.warn("Received empty WebSocket message:", event);
        return;
      }

      const data = JSON.parse(event.data);
      console.log("Message received:", data);
      if (data.function === "critic_answer") {
        if (!data.message) {
          data.message = data.result;
        }
        if (!data.agent_name) {
          data.agent_name = currentAgent;
        }
      }

      // Assign client_id if provided
      if (data.client_id) {
        localStorage.setItem("client_id", data.client_id);
        console.log("Assigned Client ID:", data.client_id);
        return;
      }

      // Handle pending function call responses
      if (data.request_id && pendingCallbacks[data.request_id]) {
        pendingCallbacks[data.request_id](data.result);
        delete pendingCallbacks[data.request_id];
        return;
      }

      // Display message
      displayMessage(data);

      // Manage input field visibility based on message content
      if (
        data.input &&
        ![
          "Choose an action",
          "Do you want to edit the code",
          "CODE SELECTION Please select the code to keep",
        ].some((text) => data.message?.includes(text))
      ) {
        isInputRequired = false;
        displayInputField(); // Show input area
        $(".buffering-circle").hide();
      } else {
        isInputRequired = false;
        hideInputField(); // Hide input area
      }
      if (
        data.message?.includes("(yes/no)") ||
        data.message?.includes("(E/EXIT)") ||
        data.message?.includes("(Y/YES)") ||
        data.message?.includes("(N/NO)")
      ) {
        isInputRequired = true;
        displayInputField();
      } else {
        isInputRequired = false;
      }

      // If input is required or certain messages appear, remove buffering animation
      if (data.input || data.message?.includes("Choose an action")) {
        $(".buffering-circle").hide();
      }
    } catch (e) {
      console.error("Error parsing WebSocket message:", e);
      console.debug("Raw event data:", event);
    }
  }

  reconnect() {
    const retryDelay = 5000; // Fixed interval of 5 seconds

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn(
        "Max reconnection attempts reached. Please check the WebSocket server."
      );
      let statusDiv = document.getElementById("connection-status");
      if (statusDiv) {
        statusDiv.style.display = "block";
        statusDiv.innerText =
          "Unable to reconnect. Please check the WebSocket server.";
      }
      return; // Stops retrying after max attempts
    }

    console.warn(
      `Reconnecting WebSocket (attempt ${this.reconnectAttempts + 1}) in ${
        retryDelay / 1000
      } seconds...`
    );

    this.reconnectAttempts++;
    let statusDiv = document.getElementById("connection-status");
    if (statusDiv) {
      statusDiv.style.display = "block";
      statusDiv.innerText = `Connection attempt ${this.reconnectAttempts}`;
    }

    setTimeout(() => this.initWebSocket(), retryDelay);
  }

  handelError() {
    // Show error in the connection status layer
    let statusDiv = document.getElementById("connection-status");
    if (statusDiv) {
      statusDiv.style.display = "block";
      statusDiv.innerText = "WebSocket error occurred. Reconnecting...";
    }
    // Ensure the socket is closed before attempting reconnection
    if (this.socket.readyState !== WebSocket.CLOSED) {
      this.socket.close();
    }
  }
}

setTimeout(() => {
  // Get WebSocket URL from URL parameters if available
  wsUrl = getUrlParameter("wsurl") || wsUrl;

  uniqueId = getUniqueIdentifier();
  console.log("Identifiant Unique de cet appareil :", uniqueId);

  // Ask user for a WebSocket URL if needed
  let userInput = prompt(
    "Enter WebSocket URL and authentication token if necessary:",
    wsUrl
  );

  // Validate WebSocket URL format
  if (userInput && isValidWebSocketUrl(userInput)) {
    wsUrl = userInput;
  } else if (userInput) {
    alert("Invalid WebSocket URL. Using default.");
  }
  // Initialize WebSocket handler with final wsUrl
  wsHandler = new WebSocketHandler(wsUrl);

  // Ajouter un écouteur pour les messages WebSocket sans écraser les existants
  wsHandler.socket.addEventListener("message", function (event) {
    if (logEditorInstance) {
      addLogLine(event.data);
    } else {
      console.warn("Log not initialized.");
    }
  });
}, 1000); // Delay prompt slightly to avoid blocking

function sendWebSocketMessage(message) {
  if (!clientId) {
    // Try to get client ID from localStorage (if available)
    clientId = localStorage.getItem("client_id");

    if (!clientId) {
      console.error("Client ID not set! Cannot send message:", message);
      return;
    }
  }

  // Check if WebSocket is open before sending
  if (!wsHandler.socket || wsHandler.socket.readyState !== WebSocket.OPEN) {
    console.error("WebSocket is not connected! Message not sent:", message);
    return;
  }

  try {
    wsHandler.socket.send(JSON.stringify(message));
    console.log("Sent WebSocket message:", message);
  } catch (error) {
    console.error("Error sending WebSocket message:", error);
  }
}

function sendFunctionCall(agentName, functionName, params, callback = null) {
  const requestId = "_" + Math.random().toString(36).slice(2, 11);
  const isParallelFunction = parallelFunctions.includes(functionName);

  if (!isParallelFunction && Object.keys(activeFunctionCalls).length > 0) {
    console.warn(
      `Function call in progress: ${Object.keys(activeFunctionCalls)}`
    );
    functionCallQueue.push(() =>
      sendFunctionCall(agentName, functionName, params, callback)
    );
    return;
  }

  if (!isParallelFunction) {
    activeFunctionCalls[requestId] = functionName;
  }

  const message = {
    agent_name: agentName,
    function: functionName,
    params,
    request_id: requestId,
  };
  sendWebSocketMessage(message);

  if (callback) {
    pendingCallbacks[requestId] = (result) => {
      callback(result);
      delete activeFunctionCalls[requestId];
      if (functionCallQueue.length > 0) functionCallQueue.shift()();
    };

    setTimeout(() => {
      if (pendingCallbacks[requestId]) {
        console.warn(`Request ${requestId} timed out.`);
        delete pendingCallbacks[requestId];
        delete activeFunctionCalls[requestId];
        if (functionCallQueue.length > 0) functionCallQueue.shift()();
      }
    }, 10000);
  } else {
    delete activeFunctionCalls[requestId];
    if (functionCallQueue.length > 0) functionCallQueue.shift()();
  }
}

$(document).on("click", ".toggle-code-result", function (e) {
  e.preventDefault(); // Prevent the default link behavior
  // Check if the sibling exists
  const codeContent = $(this).siblings(".code-content");
  if (codeContent.length) {
    codeContent.toggle(); // Toggle the visibility of the content
    console.log("Toggled code content result visibility.");
  } else {
    console.error("Code content not found.");
  }
});

require.config({
  paths: {
    vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.33.0/min/vs",
  },
});

var monacoLoaderPromise = new Promise(function (resolve, reject) {
  require(["vs/editor/editor.main"], function () {
    resolve();
  });
});

const editors = [];
let activeEditor = null;
let activeSelection = null;

// Use WeakMap to associate editors with disposables
const editorDisposables = new WeakMap();

// Function to create and set up a Monaco editor with auto-save and annotation handling
async function createMonacoEditor(
  elementId,
  content,
  language,
  agentName,
  columnId,
  readOnly = false
) {
  await monacoLoaderPromise;
  const editor = monaco.editor.create(document.getElementById(elementId), {
    value: content,
    language: language,
    readOnly: readOnly, // Set to false to allow editing
    automaticLayout: true,
    wordWrap: "on",
  });
  // Initialize versions with the initial content
  let versions = [content];
  let isSettingContent = false; // Flag to prevent infinite loops
  // Create a container for controls
  const editorContainer = document.getElementById(elementId);
  const controlsContainer = document.createElement("div");
  controlsContainer.className = "editor-controls";
  // Create Undo button
  const undoButton = document.createElement("button");
  undoButton.innerText = "Undo";
  undoButton.addEventListener("click", function () {
    editor.trigger("keyboard", "undo", null);
  });
  // Create Redo button
  const redoButton = document.createElement("button");
  redoButton.innerText = "Redo";
  redoButton.addEventListener("click", function () {
    editor.trigger("keyboard", "redo", null);
  });
  // Create History selector
  const historySelect = document.createElement("select");
  const initialOption = document.createElement("option");
  initialOption.value = 0;
  initialOption.text = "Version 1";
  historySelect.appendChild(initialOption);
  // Select the last version by default
  historySelect.selectedIndex = versions.length - 1;
  historySelect.addEventListener("change", function () {
    const selectedIndex = this.selectedIndex;
    if (selectedIndex >= 0) {
      const versionContent = versions[selectedIndex];
      isSettingContent = true;
      editor.setValue(versionContent);
      isSettingContent = false;
    }
  });
  // Append controls
  controlsContainer.appendChild(undoButton);
  controlsContainer.appendChild(redoButton);
  controlsContainer.appendChild(historySelect);
  editorContainer.parentNode.insertBefore(controlsContainer, editorContainer);
  // Debounce timer variables
  let debounceSaveTimer;
  let debounceVersionTimer;
  const contentDisposable = editor.onDidChangeModelContent(function (event) {
    if (!isSettingContent) {
      // Debounce versioning: create a new version after 1000ms of inactivity
      clearTimeout(debounceVersionTimer);
      debounceVersionTimer = setTimeout(function () {
        let currentContent = editor.getValue();
        versions.push(currentContent);

        // Update historySelect options
        const option = document.createElement("option");
        option.value = versions.length - 1;
        option.text = "Version " + versions.length;
        historySelect.appendChild(option);

        // Select the last version by default
        historySelect.selectedIndex = versions.length - 1;
      }, 1000);
    }

    // Existing auto-save logic
    document.querySelector(`#save-${agentName}-${columnId}`).style.display =
      "inline-block";
    clearTimeout(debounceSaveTimer);
    debounceSaveTimer = setTimeout(function () {
      let currentContent_1 = editor.getValue();
      let requestData = { answer: currentContent_1, column_id: columnId };
      sendFunctionCall(
        agentName,
        "update_answer",
        requestData,
        function (responseMessage) {
          console.log("Auto-save response:", responseMessage);
          document.querySelector(
            `#save-${agentName}-${columnId}`
          ).style.display = "none";
        }
      );
    }, 2500);
  });
  // Annotation Handling: Listen for cursor selection changes
  const selectionDisposable = editor.onDidChangeCursorSelection(function (
    event_1
  ) {
    if (!document.getElementById("show-annotation-menu").checked) return;

    const selection = editor.getSelection();
    const selectedText = editor.getModel().getValueInRange(selection);

    if (selectedText) {
      // Get the position for the annotation menu
      const position = editor.getScrolledVisiblePosition(
        selection.getPosition()
      );
      if (position) {
        // Get editor DOM node position
        const editorDomNode = editor.getDomNode();
        const editorRect = editorDomNode.getBoundingClientRect();
        const menuTop = editorRect.top + position.top + window.scrollY;
        const menuLeft = editorRect.left + position.left + window.scrollX;

        // Make the annotation menu draggable
        $(".annotation-menu").draggable();

        // Show the annotation menu at the calculated position
        $(".annotation-menu")
          .css({ top: menuTop + 20 + "px", left: menuLeft + "px" })
          .show();

        // Update global active editor and selection
        activeEditor = editor;
        activeSelection = selection;
      }
    } else {
      // Hide the annotation menu if no text is selected
      $(".annotation-menu").hide();
    }
  });
  // Store the editor and its disposables in the global array
  editors.push({
    editor,
    disposables: [contentDisposable, selectionDisposable],
    elementId,
    versions,
    historySelect,
    isSettingContent,
  });
  return editor;
}

// Dispose of an editor and its listener
function disposeEditor(editor) {
  const disposable = editorDisposables.get(editor);
  if (disposable) {
    disposable.dispose();
    editorDisposables.delete(editor);
  }
  editor.dispose();
  // Remove from global editors array if stored
  const index = editors.indexOf(editor);
  if (index !== -1) {
    editors.splice(index, 1);
  }
}

// Function to trigger layout update for all editors
function updateEditorLayouts() {
  editors.forEach((editorObj) => {
    editorObj.editor.layout(); // Trigger layout recalculation
  });
  console.log("Editors layout updated");
}

// Add a resize event listener to handle window resizing
window.addEventListener("resize", updateEditorLayouts);

$(document).on("click", ".delete", function () {
  const column = $(this).closest(".solution-column");
  const columnId = column.attr("id"); // e.g., column-agentname-id

  // Find the editor associated with this column
  const editorObjIndex = editors.findIndex((e) => {
    const editorContainer = e.editor.getContainerDomNode();
    return editorContainer.parentNode.id === columnId;
  });

  if (editorObjIndex !== -1) {
    const editorObj = editors[editorObjIndex];
    // Dispose of the event listener
    editorObj.disposable.dispose();
    // Dispose of the editor instance
    editorObj.editor.dispose();
    // Remove from the editors array
    editors.splice(editorObjIndex, 1);
  }

  // Remove the column from the DOM
  column.remove();

  // Clear global active editor and selection if it was the one being deleted
  if (
    activeEditor &&
    activeEditor.getContainerDomNode().parentNode.id === columnId
  ) {
    activeEditor = null;
    activeSelection = null;
  }
});
