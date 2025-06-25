function displayMessage(data) {
  if (data.message_type === "USER_ID") {
    SendUserIdOnWebsocket();
    return;
  }
  if (data.agent_name == null) {
    return;
  }
  setCurrentLLMInUse(data);
  setCurrentAgent(data);

  if (data.optional)
    if (!document.getElementById("show-optionals").checked) {
      return;
    }

    let title = `${data.agent_name} | ${data.message_type
  } | ${new Date().toLocaleTimeString()}`;
  if (data.message_type === null) {
    title = `${data.agent_name} | ${new Date().toLocaleTimeString()}`;
  }
  // Initialize content variable
  let content = "";

  if (
    ("message" in data &&
      data.message?.includes("Time spent in each option and occurrences")) ||
    data.message?.includes("inference results received") ||
    data.message?.includes("Choose an action (or hit Enter for inference)")
  ) {
    return;
  }
  SetInferenceStreamingOutput(data);
  if (data.message_type === null) {
    data.message_type = " ";
  }
  if (data.message_type === "time_end") {
    // Convert remaining time to minutes and seconds
    handelTimeEnd(data);
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
    updateDataInColumns(data);
    return;
  }

  updateDataInColumnsForTaskIdentificationAgent(data);

  if (
    data.column_max !== undefined &&
    data.column_id !== undefined &&
    data.message_type === "NEW inference result received"
  ) {
    updateNewInferenceResult(data);
  } else if (data.message_type === "successful_tasks_list") {
    updateSuccessfullTaskList(data);
  } else if (
    data.message.includes(
      "Time spent in each option and occurrences" ||
        data.message.includes("Choose an action (or hit Enter for inference)")
    )
  ) {
    console.log("do nothing");
  } else if (
    data.message.includes(
      "Do you want to reset the environment for searching a new task"
    )
  ) {
    resetEnvForNewTask();
  } else if (data.message_type === "Scores") {
    let $continueButton = $(`#continue-button-${data.agent_name}`);
    $continueButton.prop("disabled", false);

    handleEvaluationResults(data);
  } else if (data.message === "Skip for how many rounds? ") {
    skipNumberOfRounds();
  } else if (data.message_type === "NUM_PARALLEL_INFERENCES") {
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
  } else if (data.message_type == "INFERENCE CHOICE") {
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
  } else if (
    data.message_type === "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE"
  ) {
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
    data.message_type?.includes("Inference streaming output") &&
    (data.agent_name === "TaskIdentificationAgent" ||
      data.agent_name === "PlannerAgent")
  ) {
    // Extract the inferenceNumber
    let inferenceNumber = parseInt(data.message_type?.slice(-1));
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
    displayMenuOptions(data);
  } else if (
    data.agent_name === "CONFIG" &&
    data.message.includes("Enter a capital letter for subdirectory")
  ) {
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
    const cleanSuggestions = data.message;
    let suggestions = JSON.parse(cleanSuggestions);
    annotationswithID = suggestions.suggestions;
    let comment_editor =
      window[`editorInstance_${data.agent_name}-${suggestions.output_id}`];
    // Add text in the comment editor
    comment_editor.setValue(annotationswithID);
  } else if (
    data.message.includes("Enter the number of the new default LLM (0-3):")
  ) {
    // Send the selectedLLM to the backend
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
    showMessageForPlannerAgent(title, content);
  } else {
    if ($agentAccordion.length === 0) {
      $agentAccordion = $(
        `<div data-agent="${data.agent_name}"><h3>${data.agent_name}</h3><div></div></div>`
      );
      $("#message-accordion").append($agentAccordion);
    }

    let editorId = `editor-${Date.now()}`;

    let $messageContainer = $(
      `<h4>${title}</h4><div class="column-container"><div id="${editorId}"></div></div>`
    );

    $agentAccordion.children("div").append($messageContainer);
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
