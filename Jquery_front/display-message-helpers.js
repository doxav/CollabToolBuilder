function SendUserIdOnWebsocket() {
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
}

function setCurrentLLMInUse(data) {
  if (
    data.agent_name !== currentAgent &&
    data.message_type !== "orchestrate_agents" &&
    data.message_type !== "null"
  ) {
    current_llm_in_use = "default_llm";
  }
}

function setCurrentAgent(data) {
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
}

function SetInferenceStreamingOutput(data) {
  // Check if message_type includes "Inference streaming output"
  if (
    "message" in data &&
    data.message_type &&
    data.message_type?.includes("Inference streaming output")
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
                        <span class="arrow"><i class="fa-solid fa-chevron-down"></i></span> <span>${isCode ? "Code" : "Prompt"
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
}

function handelTimeEnd(data) {
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
}

function updateDataInColumns(data) {
  // Retrieve the columns of the CodingAgent
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
      console.log(`Monaco editor content updated for agent editor-${columnId}`);
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
    let errorContent = targetColumn.querySelector(`#error-content-${columnId}`);
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
    let errorContent = targetColumn.querySelector(`#error-content-${columnId}`);
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
}

function updateDataInColumnsForTaskIdentificationAgent(data) {
  if (
    data.agent_name === "TaskIdentificationAgent" &&
    (data.message_type?.includes("MULTIPLE inferences received") ||
      data.message_type?.includes("AFTER"))
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
}

function updateNewInferenceResult(data) {
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
}

function updateSuccessfullTaskList(data) {
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
}
function resetEnvForNewTask() {
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
}
function skipNumberOfRounds() {
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
}

function showMessageForPlannerAgent(title, content) {
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
}