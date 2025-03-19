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
        ].some((text) => data.message.includes(text))
      ) {
        isInputRequired = false;
        displayInputField(); // Show input area
        $(".buffering-circle").hide();
      } else {
        isInputRequired = false;
        hideInputField(); // Hide input area
      }
      if (
        data.message.includes("(yes/no)") ||
        data.message.includes("(E/EXIT)") ||
        data.message.includes("(Y/YES)") ||
        data.message.includes("(N/NO)")
      ) {
        isInputRequired = true;
        displayInputField();
      } else {
        isInputRequired = false;
      }

      // If input is required or certain messages appear, remove buffering animation
      if (data.input || data.message.includes("Choose an action")) {
        $(".buffering-circle").hide();
      }
    } catch (e) {
      console.error("Error parsing WebSocket message:", e);
      console.debug("Raw event data:", event);
    }
  }

  reconnect() {
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

    const retryDelay = Math.min(3000 * this.reconnectAttempts, 10000); // Exponential backoff (max 10s)
    console.warn(
      `Reconnecting WebSocket (attempt ${this.reconnectAttempts + 1}) in ${
        retryDelay / 1000
      } seconds...`
    );

    this.reconnectAttempts++;
    let statusDiv = document.getElementById("connection-status");
    if (statusDiv) {
      statusDiv.style.display = "block";
      statusDiv.innerText = `Connection attempt ${this.reconnectAttempts} in ${
        retryDelay / 1000
      } seconds...`;
    }

    setTimeout(() => this.initWebSocket(), retryDelay);
  }

  handelError() {
    console.error("WebSocket error:", error);

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
    wsHandler.socket.send(
      JSON.stringify({
        client_id: clientId,
        ...message,
      })
    );
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

function displayMessage(data) {
  if (data.message_type === "USER_ID") {
    // Send user_id on websocket connection
    console.log("Sending user_id to the server");
    // Wait a second before sending the user_id
    setTimeout(function () {
      sendWebSocketMessage({ message: uniqueId });
      unique_id_sent = true;
    }, 1000);
    inputAwaited = false; // Reset input flag
    isInputRequired = false;
    hideInputField(); // Hide input area after sending message
    return;
  }

  if (data.agent_name == null) {
    return;
  }
  if (
    data.agent_name !== currentAgent &&
    data.message_type !== "orchestrate_agents" &&
    data.message_type !== "null"
  ) {
    console.log(
      "Current agent is : ",
      currentAgent,
      " and data.agent_name is : ",
      data.agent_name
    );
    current_llm_in_use = "default_llm";
  }

  if (data.agent_name !== undefined) {
    if (currentAgent !== data.agent_name) {
      // Clear the agent content for this agent
      let $agentAccordion = $(
        `#message-accordion > div[data-agent="${data.agent_name}"]`
      );
      if ($agentAccordion.length) {
        $agentAccordion.children("div").empty();
      }
    }
    currentAgent = data.agent_name;
  }
  if (data.optional)
    if (!document.getElementById("show-optionals").checked) {
      return;
    }

  let title = `${data.agent_name} | ${
    data.message_type
  } | ${new Date().toLocaleTimeString()}`;
  if (data.message_type === null) {
    title = `${data.agent_name} | ${new Date().toLocaleTimeString()}`;
  }
  // Initialize content variable
  let content = "";

  if (
    ("message" in data &&
      data.message.includes("Time spent in each option and occurrences")) ||
    data.message.includes("inference results received") ||
    data.message.includes("Choose an action (or hit Enter for inference)")
  ) {
    // Exit the function
    // console.log('NOT Displaying message:', data);
    return;
  }
  // else console.log('Displaying message:', data);

  // Check if message_type includes "Inference streaming output"
  if (
    "message" in data &&
    data.message_type &&
    data.message_type.includes("Inference streaming output")
  ) {
    // Do not wrap the message in collapsible container
    content = `${formatMessage(data.message)}`;
  } else {
    // First, check if message is code or prompt
    let isCode = isPythonCode(data.message);
    let isPrompt = isPromptMessage(data.message);

    if (isCode || isPrompt) {
      // Wrap the message in a collapsible div with an arrow
      content = `
              <div class="collapsible-message open">
                  <div class="collapsible-header">
                      <span class="arrow"><i class="fa-solid fa-chevron-down"></i></span> <span>${
                        isCode ? "Code" : "Prompt"
                      }</span>
                  </div>
                  <div class="collapsible-content">
                      ${formatMessage(data.message)}
                  </div>
              </div>
          `;
    } else {
      content = `${formatMessage(data.message)}`;
    }
  }

  // If the message_type is null, set it to " "
  if (data.message_type === null) {
    data.message_type = " ";
  }

  if (data.message_type === "time_end") {
    // Convert remaining time to minutes and seconds
    let remainingTime = parseInt(data.message);
    let minutes = Math.floor(remainingTime / 60);
    let seconds = remainingTime % 60;
    document.getElementById(
      "remaining-time"
    ).textContent = `${minutes}m ${seconds}s`;
    // Countdown the remaining time
    let countdown = setInterval(function () {
      remainingTime--;
      minutes = Math.floor(remainingTime / 60);
      seconds = remainingTime % 60;
      document.getElementById(
        "remaining-time"
      ).textContent = `${minutes}m ${seconds}s`;
      if (remainingTime <= 0) {
        clearInterval(countdown);
      }
    }, 1000);
    // Show the task-list and update the task list
    $("#task-selection").show();
    $("#search-task-toggle").show();
    if (!init_task_list) {
      console.log("Initializing task list...");
      updateTaskList("None");
      init_task_list = true;
    }
    return;
  }

  let $agentAccordion = $(
    `#message-accordion > div[data-agent="${data.agent_name}"]`
  );

  // Places infos for each answers in columns in their columns

  if (
    (data.agent_name === "CodingAgent" || data.agent_name === "PlannerAgent") &&
    (data.message_type === "code_task_and_run_test SystemMessage" ||
      data.message_type === "fix_error" ||
      data.message_type === "CODE_RESULT" ||
      data.message_type === "UPDATED_CODE")
  ) {
    // Retrieve the columns of the CodingAgent
    console.log("AAAAAAAAAAAAAAAAAAAAAAA");
    let $agentAccordion = $(
      `#message-accordion > div[data-agent="${data.agent_name}"]`
    );
    let $columns = $agentAccordion.find(".solution-column");
    const columnId = `${data.agent_name}-${data.column_id}`;
    // Retrieve the targeted column
    let targetColumn = document.getElementById(`column-${columnId}`);

    // Check if the message_type is UPDATED_CODE and update the Monaco editor's content
    if (
      data.message_type === "UPDATED_CODE" &&
      data.agent_name &&
      data.column_id !== undefined
    ) {
      const editorContainer = document.getElementById(`column-${columnId}`);
      const editorInstance = editorContainer.editorInstance;
      if (editorContainer && editorContainer.editorInstance) {
        editorInstance.setValue(data.message);
        console.log(
          `Monaco editor content updated for agent editor-${columnId}`
        );
      } else
        console.error(
          `Monaco editor not found for agent editor-${columnId} editorContainer:${editorContainer} editorInstance:${editorInstance}/ `
        );
    }
    if (data.message_type === "fix_error") {
      // if Edition in VSCode, alert and do not display the buttons
      if (data.message.startsWith("Please edit and save")) {
        alert(data.message);
        return;
      }
      // Create a question with a Yes, No and Try autofix with LLM buttons and append it to the "error-conten-${columnId}" div
      let $question = $(`<p>Do you want to fix the error ?</p>`);
      let $noButton = $(
        `<button id="no-button">No (edit > save > continue to test it)</button>`
      );
      let $autofixButton = $(
        `<button id="autofix-button">Try autofix with LLM</button>`
      );
      let $customAutofixButton = $(
        `<button id="custom-autofix-button">Add instruction to autofix with LLM</button>`
      );
      let $container = $(`<div style="background-color: #ff7f7f;"></div>`);
      $container.append($question);
      $container.append($noButton);
      $container.append($autofixButton);
      $container.append($customAutofixButton);
      // Append container to the "error-content-${columnId}" div
      let errorContent = targetColumn.querySelector(
        `#error-content-${columnId}`
      );
      errorContent.appendChild($container[0]);
      $noButton.click(function () {
        sendWebSocketMessage({ message: "no" });
        $container.remove();
      });
      $autofixButton.click(function () {
        sendWebSocketMessage({ message: "a" });
        $container.remove();
      });
      $customAutofixButton.click(function () {
        let instruction = prompt(
          "Please enter your instruction for the LLM autofix:"
        );
        if (instruction !== null && instruction.trim() !== "") {
          sendWebSocketMessage({ message: instruction });
          $container.remove();
        } else {
          alert("No instruction provided.");
        }
      });
    } else {
      // Adds the message to the "error-content-${columnId}" div
      let errorContent = targetColumn.querySelector(
        `#error-content-${columnId}`
      );
      if (data.message_type === "CODE_RESULT") {
        // Create a collapsible message section for the CODE_RESULT
        content = `
                  <div class="code-result">
                      <a href="#" class="toggle-code-result" style="text-align: center; margin: 0; padding: 0;">[SEE UPDATED TEST DOC CONTENT]</a>
                      <pre class="code-content" style="display: none; text-align: left; margin: 0; padding: 0;">
                          ${formatMessage(data.message)}
                      </pre>
                  </div>
              `;
      } else {
        content =
          "<pre style='text-align: left; margin: 0; padding: 0;'>" +
          formatMessage(data.message) +
          "</pre>";
      }
      errorContent.innerHTML += content;
      // Adds a newlines to separate the messages
    }
    return;
  }

  if (
    data.agent_name === "TaskIdentificationAgent" &&
    (data.message_type.includes("MULTIPLE inferences received") ||
      data.message_type.includes("AFTER"))
  ) {
    // Retrieve the columns of the TaskIdentificationAgent
    let $agentAccordion = $(
      `#message-accordion > div[data-agent="${data.agent_name}"]`
    );
    let $columns = $agentAccordion.find(".solution-column");
    // Replace the content of message content for all the columns with a formated markdown message
    $columns.each(function (index, column) {
      let $message = $(column).find(".message");
      // Replace the actual content of the message with the actual content but formated with formlatMarkdown
      if ($message && $message.html())
        $message.html(formatMarkdown($message.html()));
      else
        console.log(
          `No message or no html content for agent ${data.agent_name} column $(index)`
        );
    });
  }

  // Generalize handling of multiple solutions
  if (
    data.column_max !== undefined &&
    data.column_id !== undefined &&
    data.message_type === "NEW inference result recieved"
  ) {
    console.log("BBBBBBBBBBB");
    // Wait 2 seconds for the last streaming output being received
    setTimeout(function () {
      // Do nothing
    }, 2000);
    number_created_functions++;
    document.getElementById("number-functions").textContent =
      number_created_functions;
    console.log("Handling multiple solutions");
    handleMultipleSolutions(data);
    // If it's the last solution and there are more than 4 solutions, add pagination
    if (data.column_id === data.column_max - 1 && data.column_max > 4) {
      // Add the html code for the pagination
      let pagination = document.createElement("div");
      pagination.className = "pagination";
      pagination.id = "pagination";
      pagination.innerHTML = `<button id="prev-page">Previous</button><span id="page-info">Page 1 of 1</span><button id="next-page">Next</button>`;

      // Add the pagination to the solution columns
      let solutionColumns = document.querySelector(".solution-columns");
      solutionColumns.parentNode.insertBefore(
        pagination,
        solutionColumns.nextSibling
      );

      // Add event listeners for the pagination buttons taking into account that there are 4 columns per page
      let columnMax = data.column_max;
      let currentPage = 1;
      let totalPages = Math.ceil(columnMax / 4);
      let $columns = $(".solution-column");
      let $pageInfo = $("#page-info");

      $("#next-page").click(function () {
        if (currentPage < totalPages) {
          currentPage++;
          $columns.hide();
          $columns.slice((currentPage - 1) * 4, currentPage * 4).show();
          $pageInfo.text(`Page ${currentPage} of ${totalPages}`);
        }
      });

      $("#prev-page").click(function () {
        if (currentPage > 1) {
          currentPage--;
          $columns.hide();
          $columns.slice((currentPage - 1) * 4, currentPage * 4).show();
          $pageInfo.text(`Page ${currentPage} of ${totalPages}`);
        }
      });

      $columns.hide();
      $columns.slice(0, 4).show();
      $pageInfo.text(`Page ${currentPage} of ${totalPages}`);

      // Update the pagination info with the total number of pages
      totalPages = Math.ceil(columnMax / 4);
      $pageInfo.text(`Page ${currentPage} of ${totalPages}`);
    }
  } else if (data.message_type === "successful_tasks_list") {
    console.log(data.message);
    // Step 1: Parse the outer JSON array
    const outerArray = JSON.parse(data.message);

    // Step 2: Parse each string in the array to convert it into a JSON object
    const parsedObjects = outerArray.map((item) => JSON.parse(item));

    // Step 3: Extract class_name and parameters from each function
    const extractedData = parsedObjects.map((entry) => {
      const className = entry.class_name;
      const programCode = entry.program_code;

      // Extract parameters from the program code using a regular expression
      const paramsMatch = programCode.match(/def\s+\w+\(([^)]*)\)/);
      const params = paramsMatch
        ? paramsMatch[1].split(",").map((param) => param.trim())
        : [];

      // For each param in params, removes the default value if the value exceeds 10 characters or if the default value is ""
      params.forEach((param, index) => {
        if (param.includes("=")) {
          let [paramName, paramValue] = param.split("=");
          console.log("ParamValue: ", paramValue);
          if (
            paramValue.length > 10 ||
            paramValue === '""' ||
            paramValue === '" "' ||
            paramValue === ' ""' ||
            paramValue === "None"
          ) {
            params[index] = paramName;
          }
        }
      });

      // Extract the docstring using a regular expression
      const docstringMatch = programCode.match(/"""\s*([\s\S]*?)\s*"""/);
      const docstring = docstringMatch ? docstringMatch[1].trim() : "";

      return {
        class_name: className,
        parameters: params,
        docstring: docstring,
      };
    });

    // Step 4: Extract the class names adding there parameters
    const classNames = extractedData.map((entry) => {
      const params =
        entry.parameters.length > 0 ? `(${entry.parameters.join(", ")})` : "";
      return `${entry.class_name}${params}`;
    });

    // Remove redundant class names
    const uniqueClassNames = [...new Set(classNames)];

    // For each className in classNames, add the name in a div in the func-names-content div
    const funcNamesContent = document.getElementById("func-names-content");
    funcNamesContent.innerHTML = "";
    // Create a foldable div element for each className which is folded by default and contains the docstring when unfolded
    uniqueClassNames.forEach((className, index) => {
      const foldableDiv = document.createElement("div");
      foldableDiv.className = "foldable";
      foldableDiv.innerHTML = `
              <div class="foldable-header">
                  <span class="arrow"><i class="fa-solid fa-chevron-right"></i></span>
                  <span><strong>${className}</strong></span>
              </div>
              <div class="foldable-content" style="display: none;">
                  <p>${extractedData[index].docstring}</p>
              </div>
          `;
      foldableDiv
        .querySelector(".foldable-header")
        .addEventListener("click", function () {
          const arrow = this.querySelector(".arrow i");
          const content = this.nextElementSibling;
          if (content.style.display === "none") {
            content.style.display = "block";
            arrow.classList.remove("fa-chevron-right");
            arrow.classList.add("fa-chevron-down");
          } else {
            content.style.display = "none";
            arrow.classList.remove("fa-chevron-down");
            arrow.classList.add("fa-chevron-right");
          }
        });
      funcNamesContent.appendChild(foldableDiv);
    });
  } else if (
    data.message.includes(
      "Time spent in each option and occurrences" ||
        data.message.includes("Choose an action (or hit Enter for inference)")
    )
  ) {
    //        } else if(data.message && (data.message.includes("Time spent in each option and occurrences" || data.message.includes("Choose an action (or hit Enter for inference)")))){
  } else if (
    data.message.includes(
      "Do you want to reset the environment for searching a new task"
    )
  ) {
    //        } else if (data.message && data.message.includes("Do you want to reset the environment for searching a new task")){
    // Hide the input area
    $(".input-area").hide();
    // Delete all agent accordions
    $("#message-accordion").empty();

    // Show the question to reset the environment
    let $question = $(
      `<p>Do you want to reset the environment for searching a new task (Y/YES) or search a new task by keeping what has been created by this task (N/NO/Enter) ? or just exit (E/EXIT) ?</p>`
    );
    let $yesButton = $(`<button id="yes-button">Yes</button>`);
    let $noButton = $(`<button id="no-button">No</button>`);
    let $exitButton = $(`<button id="exit-button">Exit</button>`);
    let $container = $(`<div style="font-size: large;"></div>`);
    $container.append($question);
    $container.append($yesButton);
    $container.append($noButton);
    $container.append($exitButton);
    // Ajouter le conteneur sous l'accordéon
    $("#message-accordion").append($container);
    // Gérer l'événement du bouton
    $yesButton.click(function () {
      sendWebSocketMessage({ message: "yes" });
      $container.remove();
      // Reset the number of created functions
      number_created_functions = 0;
      document.getElementById("number-functions").textContent =
        number_created_functions;
    });
    $noButton.click(function () {
      sendWebSocketMessage({ message: "no" });
      $container.remove();
    });
    $exitButton.click(function () {
      sendWebSocketMessage({ message: "exit" });
      $container.remove();
    });
  } else if (data.message_type === "Scores") {
    // Enable continue-button-${data.agent_name}
    let $continueButton = $(`#continue-button-${data.agent_name}`);
    $continueButton.prop("disabled", false);

    handleEvaluationResults(data);
    let $scoreButtons = $agentAccordion.find("button[id^='score-']");
  } else if (data.message === "Skip for how many rounds? ") {
    // Existing code for handling this case
    // COPIED CODE
    let $question = $(`<p>Skip for how many rounds ?</p>`);
    // Afficher une liste déroulante de 1 à 10 avec la possibilité de saisir un autre nombre
    let $select = $(`<select id="rounds-select"></select>`);
    for (let i = 1; i <= 10; i++) {
      $select.append(`<option value="${i}">${i}</option>`);
    }
    $select.append(`<option value="other">Other</option>`);
    // Si "Other" est sélectionné, afficher un champ de saisie pour entrer le nombre de tours
    $select.change(function () {
      if ($(this).val() === "other") {
        $(this).replaceWith(
          `<input type="number" id="rounds-input" min="1" max="100" step="1" value="1">`
        );
      }
    });
    let $button = $(`<button id="send-rounds">Send</button>`);
    let $container = $(`<div></div>`);
    $container.append($question);
    $container.append($select);
    $container.append($button);
    // Ajouter le conteneur sous l'accordéon
    $("#message-accordion").append($container);
    // Gérer l'événement du bouton pour envoyer le nombre de tours au serveur
    $button.click(function () {
      let rounds;
      if ($("#rounds-select").length) {
        rounds = $("#rounds-select").val();
      } else {
        rounds = $("#rounds-input").val();
      }

      if (rounds === "other") {
        rounds = $("#rounds-input").val();
      }

      rounds = parseInt(rounds);
      if (isNaN(rounds) || rounds < 1) {
        rounds = 1;
      } else if (rounds > 100) {
        rounds = 100;
      }
      sendWebSocketMessage({ message: rounds });

      $container.remove();
      let actionDescription = `User set skip rounds to ${rounds}`;
      addHistoryEntry(actionDescription);
    });
    // END COPY
  } else if (data.message_type === "NUM_PARALLEL_INFERENCES") {
    // Existing code for handling this case
    // COPIED CODE
    let $question = $(`<p>How many parallel inferences?</p>`);
    let $select = $(`<select id="parallel-inferences-select"></select>`);
    for (let i = 1; i <= 4; i++) {
      $select.append(`<option value="${i}">${i}</option>`);
    }
    let $button = $(`<button id="send-parallel-inferences">Send</button>`);
    let $container = $(`<div></div>`);
    $container.append($question);
    $container.append($select);
    $container.append($button);
    $("#message-accordion").append($container);
    $button.click(function () {
      let parallelInferences = $("#parallel-inferences-select").val();
      sendWebSocketMessage({ message: parallelInferences });
      $container.remove();
    });
    // END COPY
  } else if (data.message_type == "INFERENCE CHOICE") {
    // Existing code for handling this case
    // COPIED CODE
    let $question = $(
      `<p>Proceed to inference? You can also proceed using a premium LLM.</p>`
    );
    let $yesButton = $(`<button id="yes-button">Yes</button>`);
    let $noButton = $(`<button id="no-button">No</button>`);
    let $premiumButton = $(
      `<button id="premium-button">Use premium LLM</button>`
    );
    let $container = $(`<div></div>`);
    $container.append($question);
    $container.append($yesButton);
    $container.append($noButton);
    $container.append($premiumButton);
    $("#message-accordion").append($container);
    $yesButton.click(function () {
      sendWebSocketMessage({ message: "y" });
      $container.remove();
    });
    $noButton.click(function () {
      sendWebSocketMessage({ message: "n" });
      $container.remove();
    });
    $premiumButton.click(function () {
      sendWebSocketMessage({ message: "p" });
      $container.remove();
    });
    // END COPY
  } else if (
    data.message_type === "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE"
  ) {
    // Existing code for handling this case
    // COPIED CODE
    let $question = $(`<p>Proceed with synthesis mode?</p>`);
    let $yesButton = $(`<button id="yes-button">Yes</button>`);
    let $noButton = $(`<button id="no-button">No</button>`);
    let $container = $(`<div></div>`);
    $container.append($question);
    $container.append($yesButton);
    $container.append($noButton);
    $("#message-accordion").append($container);
    $yesButton.click(function () {
      sendWebSocketMessage({ message: "1" });
      $container.remove();
    });
    $noButton.click(function () {
      sendWebSocketMessage({ message: "0" });
      $container.remove();
    });
    // END COPY
  } else if (
    data.message_type.includes("Inference streaming output") &&
    (data.agent_name === "TaskIdentificationAgent" ||
      data.agent_name === "PlannerAgent")
  ) {
    //        } else if(data.message_type && data.message_type.includes("Inference streaming output") && data.agent_name === "TaskIdentificationAgent") {
    // console.log("Inference streaming output");
    // Extract the inferenceNumber
    let inferenceNumber = parseInt(data.message_type.slice(-1));
    if (isNaN(inferenceNumber)) {
      inferenceNumber = 0;
      //return;
    }

    if (data.agent_name === "PlannerAgent") {
      handleMultipleSolutions(data);
      return;
    }

    const columnId = `${inferenceNumber}`;
    let targetColumn = document.getElementById(
      `column-${data.agent_name}-${columnId}`
    );

    if (!targetColumn) {
      // Column doesn't exist yet, create it
      console.log(`Creating column for inference ${inferenceNumber}`);

      // Ensure the solutionColumnsContainer exists
      let solutionColumnsContainer = document.querySelector(
        `#message-accordion > div[data-agent="${data.agent_name}"] > div > .solution-columns`
      );

      if (!solutionColumnsContainer) {
        // Create the solution columns container
        solutionColumnsContainer = document.createElement("div");
        solutionColumnsContainer.className = "solution-columns";
        let agentAccordion = document.querySelector(
          `#message-accordion > div[data-agent="${data.agent_name}"] > div`
        );
        if (!agentAccordion) {
          // Create agent accordion if it doesn't exist
          agentAccordion = document.createElement("div");
          agentAccordion.setAttribute("data-agent", data.agent_name);
          agentAccordion.innerHTML = `<h3>${data.agent_name}</h3><div></div>`;
          document
            .querySelector("#message-accordion")
            .appendChild(agentAccordion);
        }

        agentAccordion.appendChild(solutionColumnsContainer);
      }
      // Create the column
      targetColumn = document.createElement("div");
      targetColumn.className = "solution-column";
      targetColumn.id = `column-${data.agent_name}-${columnId}`;
      console.log("adding column:", targetColumn.id);
      targetColumn.innerHTML = `
              <button class="expand-button"><i class="fas fa-expand"></i></button>
              <h4>Answer ${inferenceNumber}</h4>
              <div class="message" style="max-height: 600px; overflow-y: auto;"></div>
          `;
      solutionColumnsContainer.appendChild(targetColumn);
      adjustColumnWidths();
    }

    // Append the message to the column's message div
    const messageDiv = targetColumn.querySelector(".message");
    if (messageDiv) {
      messageDiv.innerHTML += formatMessage(data.message);
      // messageDiv.scrollTop = messageDiv.scrollHeight;
    } else
      console.log(
        `No message div found for column-${data.agent_name}-${columnId}`
      );
  } else if (
    data.message_type === "BEFORE inference action MENU" ||
    (data.message_type === "AFTER inference action MENU" &&
      data.message.includes("Change default")) ||
    data.message.includes("Change premium") ||
    data.message.includes("Exit") ||
    data.message.includes("Log comments")
  ) {
    //        } else if (data.message_type === "BEFORE inference action MENU" || data.message_type === "AFTER inference action MENU" /*) && (data.message.includes("Change default") || data.message.includes("Change premium") || data.message.includes("Exit") || data.message.includes("Log comments"))*/) {
    displayMenuOptions(data);
  } else if (
    data.agent_name === "CONFIG" &&
    data.message.includes("Enter a capital letter for subdirectory")
  ) {
    // Existing code for handling this case
    // COPIED CODE
    // Show a message in a new accordion named "CONFIG"
    let $configAccordion = $(`#message-accordion > div[data-agent="CONFIG"]`);
    if ($configAccordion.length === 0) {
      $configAccordion = $(
        `<div data-agent="CONFIG"><h3>CONFIG</h3><div></div></div>`
      );
      $("#message-accordion").append($configAccordion);
    }
    let $messageContainer = $(
      `<h4>${title}</h4><div class="column-container">Please, select a subdirectory </div>`
    );
    $configAccordion.children("div").append($messageContainer);
    displayMenuOptions(data);
    // END COPY
  } else if (
    data.agent_name === "Pipeline/Function Mode" &&
    data.message_type === "Files loaded"
  ) {
    // Do nothing
  } else if (
    (data.agent_name === "TaskIdentificationAgent" ||
      data.agent_name === "CodingAgent") &&
    data.message_type === "CRITIC SUGGESTIONS"
  ) {
    console.log("CRITIC SUGGESTIONS");
    console.log(data.message);
    const cleanSuggestions = data.message;
    let suggestions = JSON.parse(cleanSuggestions);
    console.log("suggestsions:", suggestions);
    annotationswithID = suggestions.suggestions;
    console.log("output_id:", suggestions.output_id);
    let comment_editor =
      window[`editorInstance_${data.agent_name}-${suggestions.output_id}`];
    console.log("editor:", comment_editor);
    // Add text in the comment editor
    comment_editor.setValue(annotationswithID);
  } else if (
    data.message.includes("Enter the number of the new default LLM (0-3):")
  ) {
    // Send the selectedLLM to the backend
    console.log("Sending the selected LLM to the backend : ", selectedLLM);
    sendWebSocketMessage({ message: selectedLLM });
  } else if (
    data.message_type === "TASK SELECTION" ||
    data.message_type === "VALIDATION_INFO" ||
    data.message_type === "Capitalization_info" ||
    data.message_type === "ADDITIONAL_INFO"
  ) {
    if ($agentAccordion.length === 0) {
      $agentAccordion = $(
        `<div data-agent="${data.agent_name}"><h3>${data.agent_name}</h3><div></div></div>`
      );
      $("#message-accordion").append($agentAccordion);
    }

    let $messageContainer = $(
      `<h4>${title}</h4><div class="column-container"><pre>${content}</pre></div>`
    );
    $agentAccordion.children("div").append($messageContainer);

    if (data.input) {
      displayInputField();
    }
  } else if (data.agent_name === "PlannerAgent") {
    // Show message in a new accordion named "PlannerAgent" if it doesn't exist else append the message to the existing accordion
    let $plannerAccordion = $(
      `#message-accordion > div[data-agent="PlannerAgent"]`
    );
    if ($plannerAccordion.length === 0) {
      $plannerAccordion = $(
        `<div data-agent="PlannerAgent"><h3>PlannerAgent</h3><div></div></div>`
      );
      $("#message-accordion").append($plannerAccordion);
    }
    let $messageContainer = $(
      `<h4>${title}</h4><div class="column-container">${content}</div>`
    );
    $plannerAccordion.children("div").append($messageContainer);
    if (data.input) {
      displayInputField();
    }
  } else {
    if ($agentAccordion.length === 0) {
      $agentAccordion = $(
        `<div data-agent="${data.agent_name}"><h3>${data.agent_name}</h3><div></div></div>`
      );
      $("#message-accordion").append($agentAccordion);
    }

    // Generate a unique ID for the editor
    let editorId = `editor-${Date.now()}`;

    // Create the container with the editor ID
    let $messageContainer = $(
      `<h4>${title}</h4><div class="column-container"><div id="${editorId}"></div></div>`
    );

    $agentAccordion.children("div").append($messageContainer);
    // alert(data.message);
    // Initialize Monaco Editor
    //createMonacoEditor(editorId, data.message, 'markdown', data.agent_name, 0).then(function(editor) {messageContainer[0].editorInstance = editor;});

    if (data.input) {
      displayInputField();
    }
  }

  $("#message-accordion").accordion("refresh");
  updateFilters(data);

  // Apply syntax highlighting to code blocks
  $("pre code").each(function (i, block) {
    hljs.highlightBlock(block);
  });

  // Auto-open accordion if checkbox is checked
  if ($("#auto-open-accordion").is(":checked")) {
    let accordionIndex = $("#message-accordion > div").index($agentAccordion);
    $("#message-accordion").accordion("option", "active", accordionIndex);
    // Wait a second then scroll to the bottom
    setTimeout(scrollToBottom, 1000);
  }

  // Auto-open accordion if auto-open-accordion-bottom checkbox is checked
  if ($("#auto-open-accordion-bottom").is(":checked")) {
    let accordionIndex = $("#message-accordion > div").index($agentAccordion);
    $("#message-accordion").accordion("option", "active", accordionIndex);
    // Wait a second then scroll to the bottom
    setTimeout(scrollToBottom, 1000);
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
