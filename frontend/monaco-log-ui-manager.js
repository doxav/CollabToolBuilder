let messageCounter = 0;
let logEditorInstance;
let logData = ""; // Variable pour stocker les logs
let logLines = [];
let current_llm_in_use = "default_llm";

monacoLoaderPromise
  .then(function () {
    logEditorInstance = monaco.editor.create(
      document.getElementById("log-editor"),
      {
        value: "",
        language: "json",
        readOnly: true,
        automaticLayout: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        wordWrap: "on",
      }
    );
  })
  .catch(function (error) {
    console.error(
      "Erreur lors de l'initialisation de l'éditeur de log Monaco:",
      error
    );
  });

// Référence aux éléments du DOM
const logContainer = document.getElementById("log-container");
const toggleLogButton = document.getElementById("toggle-log-button");
const searchTask = document.getElementById("search-task-toggle");

toggleLogButton.addEventListener("click", function () {
  if (logContainer.style.display === "none") {
    logContainer.style.display = "block";
    toggleLogButton.textContent = "Cacher le Log";
  } else {
    logContainer.style.display = "none";
    toggleLogButton.textContent = "Afficher le Log";
  }
});

searchTask.addEventListener("input", function (event) {
  filterTaskList();
});

function filterTaskList() {
  let searchValue = searchTask.value.toLowerCase();
  $("#task-selection option").each(function () {
    let text = $(this).text().toLowerCase();
    $(this).toggle(text.includes(searchValue));
  });
}

function addLogLine(message) {
  logLines.push(message.replace(/\\n/g, "\n"));
  if (logLines.length > 10000)
    // Keep only the last 10 000 lines
    logLines.shift();
  // Update the logData
  logData = logLines.join("\n");
  if (logEditorInstance) {
    logEditorInstance.setValue(logData);
    logEditorInstance.revealLine(logEditorInstance.getModel().getLineCount());
  }
}

document.addEventListener("keydown", function (event) {
  // Check if Ctrl + Shift + L is pressed to display the log editor
  if (event.ctrlKey && event.shiftKey && event.code === "KeyL") {
    event.preventDefault(); // Prevent default action (if any)
    // Toggle the display of the log editor
    const logContainer = document.getElementById("log-container");
    if (
      logContainer.style.display === "none" ||
      logContainer.style.display === ""
    )
      logContainer.style.display = "block"; // Show the log editor
    else logContainer.style.display = "none"; // Hide the log editor
  }
});
function hideAgentButtons(agentName) {
  $(`div[data-agent="${agentName}"]`)
    .find(
      'button#autofix-button, button[id^="select-"], button[id^="delete-"], button[id^="evaluate-"], button[id^="regenwithmymodifbutton-"], button[id^="regenallbutton-"], button[id^="suggestions-"], input[id^="comment-"], button#custom-autofix-button'
    )
    .hide();
}

function showAgentButtons(agentName) {
  //$(`div[data-agent="${agentName}"]`).find('button#autofix-button, button[id^="select-"], button[id^="delete-"], button[id^="evaluate-"], button[id^="regenwithmymodifbutton-"], button[id^="regenallbutton-"], input[id^="comment-"], button#custom-autofix-button').show();
  $(`div[data-agent="${agentName}"]`)
    .find(
      'button#autofix-button, button[id^="delete-"], button[id^="evaluate-"], button[id^="regenwithmymodifbutton-"], button[id^="regenallbutton-"], button[id^="suggestions-"], input[id^="comment-"], button#custom-autofix-button'
    )
    .show();
}

// Function to update button visibility based on the selector
function updateButtonVisibility() {
  const selection = $("#button-visibility-selector").val();
  const continueButton_isVisible =
    $('.menu-option[style*="background-color: green"]').is(":visible") ||
    $(`#continue-button-${currentAgent}`).is(":visible") ||
    $(`button[id^="score-${currentAgent}"]`).is(":visible"); // continue-button-

  // Define selectors for buttons and inputs to show/hide for each session
  const session1ShowSelectors = [
    "button#autofix-button",
    //'button[id^="select-"]',
    'button[id^="delete-"]',
    // Include any other session 1 specific buttons here
  ];
  const session1HideSelectors = [
    'button[id^="suggestions-"]',
    'button[id^="evaluate-"]',
    'button[id^="regenwithmymodifbutton-"]',
    'button[id^="regenallbutton-"]',
    'input[id^="comment-"]',
    "button#custom-autofix-button",
    // Include any other buttons/inputs to hide in SESSION_1
  ];

  const session2ShowSelectors = [
    'button[id^="suggestions-"]',
    'button[id^="evaluate-"]',
    'button[id^="regenwithmymodifbutton-"]',
    'button[id^="regenallbutton-"]',
    'input[id^="comment-"]',
    "button#custom-autofix-button",
    // Include any other session 2 specific buttons or inputs here
  ];
  const session2HideSelectors = [
    "button#autofix-button",
    'button[id^="select-"]',
    'button[id^="delete-"]',
  ];
  const session3ShowSelectors = [
    "button#autofix-button",
    'button[id^="delete-"]',
    'button[id^="suggestions-"]',
    'button[id^="evaluate-"]',
    'button[id^="regenwithmymodifbutton-"]',
    'button[id^="regenallbutton-"]',
    'input[id^="comment-"]',
    "button#custom-autofix-button",
    // Include any other session 2 specific buttons or inputs here
  ];
  const session3HideSelectors = [];

  const studentShowSelectors = [
    "button#autofix-button",
    'button[id^="delete-"]',
    'button[id^="suggestions-"]',
    'button[id^="evaluate-"]',
    'button[id^="regenwithmymodifbutton-"]',
    'button[id^="regenallbutton-"]',
    'input[id^="comment-"]',
    "button#custom-autofix-button",
    // Include any other session 2 specific buttons or inputs here
  ];
  const studentHideSelectors = [];

  const studentExpertShowSelectors = [
    "button#autofix-button",
    'button[id^="delete-"]',
    'button[id^="suggestions-"]',
    'button[id^="evaluate-"]',
    'button[id^="regenwithmymodifbutton-"]',
    'button[id^="regenallbutton-"]',
    'input[id^="comment-"]',
    "button#custom-autofix-button",
    // Include any other session 2 specific buttons or inputs here
  ];
  const studentExpertHideSelectors = [];

  const TP4ShowSelectors = [
    "button#autofix-button",
    'button[id^="delete-"]',
    'button[id^="suggestions-"]',
    'button[id^="evaluate-"]',
    'input[id^="comment-"]',
    "button#custom-autofix-button",
    // Include any other session 2 specific buttons or inputs here
  ];
  const TP4HideSelectors = [
    'button[id^="regenwithmymodifbutton-"]',
    'button[id^="regenallbutton-"]',
  ];

  const TP3ShowSelectors = [
    "button#autofix-button",
    'button[id^="delete-"]',
    'button[id^="evaluate-"]',
    'input[id^="comment-"]',
    "button#custom-autofix-button",
    // Include any other session 2 specific buttons or inputs here
  ];
  const TP3HideSelectors = [
    'button[id^="regenwithmymodifbutton-"]',
    'button[id^="regenallbutton-"]',
    'button[id^="suggestions-"]',
  ];

  // First, reset any previous visibility changes
  // Show all elements that might have been hidden in previous sessions
  if (continueButton_isVisible)
    session1ShowSelectors
      .concat(
        session1HideSelectors,
        session2ShowSelectors,
        session2HideSelectors
      )
      .forEach((selector) => {
        $(selector).show();
      });
  else
    session1ShowSelectors
      .concat(
        session1HideSelectors,
        session2ShowSelectors,
        session2HideSelectors
      )
      .forEach((selector) => {
        $(selector).hide();
      });

  if (selection === "SESSION_1") {
    session1HideSelectors.forEach((selector) => {
      $(selector).hide();
    });
    if (continueButton_isVisible)
      session1ShowSelectors.forEach((selector) => {
        $(selector).show();
      });
    else $("button#autofix-button").show();
    $(".menu-option").hide();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("go back");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("edit answer");
      })
      .show();
  } else if (selection === "SESSION_2") {
    session2HideSelectors.forEach((selector) => {
      $(selector).hide();
    });
    if (continueButton_isVisible)
      session2ShowSelectors.forEach((selector) => {
        $(selector).show();
      });
    else $("button#custom-autofix-button").show();
    $(".menu-option").hide();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("go back");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("give instruction or");
      })
      .show();
  } else if (selection === "SESSION_STUDENT") {
    studentHideSelectors.forEach((selector) => {
      $(selector).hide();
    });
    if (continueButton_isVisible)
      studentShowSelectors.forEach((selector) => {
        $(selector).show();
      });
    else $("button#custom-autofix-button, button#autofix-button").show();
    $(".menu-option").hide();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("go back");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("give instruction or");
      })
      .show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('modify agent');}).show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('skip for');}).show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('set num of parallel');}).show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('inferences checks');}).show();
    //$('.menu-option').filter(function() {return $(this).attr('id') === 'llm-select';}).show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("change default agent");
      })
      .show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('critic to improve');}).show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('edit answer');}).show();
  } else if (selection === "SESSION_STUDENT_EXPERT") {
    studentExpertHideSelectors.forEach((selector) => {
      $(selector).hide();
    });
    if (continueButton_isVisible)
      studentShowSelectors.forEach((selector) => {
        $(selector).show();
      });
    else $("button#custom-autofix-button, button#autofix-button").show();
    $(".menu-option").hide();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("go back");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("give instruction or");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("modify agent");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("skip for");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("set num of parallel");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("inferences checks");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).attr("id") === "llm-select";
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("change default agent");
      })
      .show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('critic to improve');}).show();
    //$('.menu-option').filter(function() {return $(this).text().toLowerCase().includes('edit answer');}).show();
  } else if (selection === "SESSION_TP4") {
    TP4HideSelectors.forEach((selector) => {
      $(selector).hide();
    });
    if (continueButton_isVisible)
      TP4ShowSelectors.forEach((selector) => {
        $(selector).show();
      });
    else $("button#custom-autofix-button, button#autofix-button").show();
    $(".menu-option").hide();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("go back");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("give instruction or");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("change default agent");
      })
      .show();
  } else if (selection === "SESSION_TP3") {
    TP3HideSelectors.forEach((selector) => {
      $(selector).hide();
    });
    if (continueButton_isVisible)
      TP3ShowSelectors.forEach((selector) => {
        $(selector).show();
      });
    else $("button#custom-autofix-button, button#autofix-button").show();
    $(".menu-option").hide();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("go back");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("give instruction or");
      })
      .show();
    $(".menu-option")
      .filter(function () {
        return $(this).text().toLowerCase().includes("change default agent");
      })
      .show();
  } else if (selection === "ALL" || selection === "ALL_SEND") {
    $(".menu-option").show();
    $("button#autofix-button, button#custom-autofix-button").show();
    if (selection === "ALL_SEND")
      // Display input & send buttons
      $(".input-area, #user-input, #send-button").show();
  }

  // Show Continue menu-option for all sessions
  $('.menu-option[style*="background-color: green"]').show();
  const $greenContinueButton = $(
    '.menu-option[style*="background-color: green"]'
  );
  if ($greenContinueButton.length == 1) {
    // Step 2: Find the closest solution column containing this button
    const $column = $greenContinueButton.closest(".solution-column");

    if ($column.length === 0) {
      return;
    }

    // Step 3: Extract the columnId from the column's id attribute
    const fullColumnId = $column.attr("id"); // e.g., "column-CodingAgent-0"
    const columnIdParts = fullColumnId.split("-");
    if (columnIdParts.length < 3) {
      console.error(`Invalid column id format: ${fullColumnId}`);
      return;
    }
    const agentName = columnIdParts[1];
    const columnId = columnIdParts.slice(2).join("-"); // Handles cases with multiple dashes in columnId
    const fullAgentColumnId = `${agentName}-${columnId}`; // e.g., "CodingAgent-0"

    $(".solution-column")
      .not($column)
      .each(function () {
        const otherFullColumnId = $(this)
          .attr("id")
          .replace(/^column-/, ""); // Remove 'column-' prefix
        $(`button#save-${otherFullColumnId},
               button#evaluate-${otherFullColumnId},
               button#evaluate-${otherFullColumnId},
               button#regenallbutton-${otherFullColumnId},
               button#regenwithmymodifbutton-${otherFullColumnId},
               input#comment-${otherFullColumnId}`).hide();
      });
  } else if ($greenContinueButton.length > 1) {
    console.log("multiple columns with button:" + $greenContinueButton.length);
  } else {
    // If no green "Continue" button is visible, hide all relevant buttons across all columns
    $(`button[id^="save-"],
           button[id^="suggestions-"],
           button[id^="evaluate-"],
           button[id^="regenallbutton-"],
           button[id^="regenwithmymodifbutton-"],
           input[id^="comment-"]`).hide();
  }
}

// Call updateButtonVisibility when the selector changes
$("#button-visibility-selector").change(function () {
  updateButtonVisibility();
});

// Set up the periodic execution every 300ms
setInterval(updateButtonVisibility, 300);

// Global variables
let selectedSolution = [];
// Function to hide input area
function hideInputField() {
  $(".input-area").hide();
  // Show buffering circle
  $(".buffering-circle").show();
}

function getColumnsToKeep(agent) {
  console.log(1);
  let columnsToKeep = [];
  console.log(2);
  let totalColumns = document.querySelectorAll(
    `.solution-column[id^='column-${agent}-']`
  ).length;
  console.log(`.solution-column[id^='column-${agent}-']`);
  for (let i = 0; i < totalColumns; i++) {
    console.log(4);
    let col = document.getElementById(`column-${agent}-${i}`);
    console.log(5);
    if (!col.classList.contains("deleted")) columnsToKeep.push(i);
  }
  console.log(6);
  return columnsToKeep;
}

// Mapping of agent names to their evaluation types
const agentEvaluations = {
  CodingAgent: "score",
  PlannerAgent: "score",
  TaskIdentificationAgent: "compliance",
  // Add more agents and their evaluation types here
};

function formatPythonDict2Table(fullText, includeHeader = false) {
  try {
    const dataArray = JSON.parse(fullText.replace(/'/g, '"')); // Parse the JSON string

    let tableContent = "";
    if (includeHeader) {
      tableContent += "Value\tKey\n";
      tableContent += "------------------------------------------\n";
    }

    dataArray.forEach((dict) => {
      for (const [key, value] of Object.entries(dict)) {
        const formattedValue =
          typeof value === "number" && !Number.isInteger(value)
            ? value.toFixed(2)
            : value;
        tableContent += `${formattedValue}\t${key}\n`;
      }
      if (includeHeader) {
        tableContent += "------------------------------------------\n";
      }
    });

    return tableContent;
  } catch (e) {
    return `Failed to format data. Original content:\n\n${fullText}`;
  }
}

function addHistoryEntry(actionDescription) {
  let timestamp = new Date().toLocaleTimeString();
  let $entry = $(
    `<div style="font-size: small"><strong>[${timestamp}] ${currentAgent}</strong><br>${actionDescription}</div>`
  );
  $("#history-content").append($entry);
}

$(`#temperature`).on("input", function () {
  $(`#value-temp`).text(this.value);
});

function createMonacoDiffEditor(elementId, originalContent, modifiedContent) {
  return monacoLoaderPromise.then(function () {
    const diffEditor = monaco.editor.createDiffEditor(
      document.getElementById(elementId),
      {
        automaticLayout: true,
        originalEditable: false, // Keep original content read-only
        readOnly: false, // Allow editing the modified content
        wordWrap: "on",
      }
    );
    const originalModel = monaco.editor.createModel(
      originalContent,
      "markdown"
    );
    const modifiedModel = monaco.editor.createModel(
      modifiedContent,
      "markdown"
    );
    monaco.editor.setModelLanguage(modifiedModel, "markdown"); // Format modified content as markdown
    diffEditor.setModel({ original: originalModel, modified: modifiedModel });
    return diffEditor;
  });
}

function activateCodeComparison(agent_columnId, newContent) {
  // Compute the editor container's ID based on agent_columnId
  const editorContainerId = `editor-${agent_columnId}`;
  const editorContainer = document.getElementById(editorContainerId);

  if (!editorContainer) {
    console.error(`Editor container with ID ${editorContainerId} not found.`);
    enableGenButtons(true); // Re-enable buttons if editor not found
    return;
  }

  // Retrieve the existing editor instance
  const existingEditor = editorContainer.editorInstance;

  if (!existingEditor) {
    console.error(
      `No editor instance found in container ${editorContainerId}.`
    );
    enableGenButtons(true); // Re-enable buttons if editor instance not found
    return;
  }

  // Get the original content from the existing editor
  const originalContent = existingEditor.getValue();

  // Dispose of the existing editor instance to prevent stacking
  existingEditor.dispose();

  // Create the Diff Editor with original and new content
  createMonacoDiffEditor(editorContainerId, originalContent, newContent)
    .then(function (diffEditorInstance) {
      // Store the Diff Editor instance
      editorContainer.editorInstance = diffEditorInstance;

      // Check if the toggle button already exists to prevent duplicates
      if (!editorContainer.querySelector(".toggle-diff-button")) {
        // Create the toggle button
        const toggleButton = document.createElement("button");
        toggleButton.textContent = "Toggle Diff";
        toggleButton.className = "toggle-diff-button";
        toggleButton.style.position = "absolute";
        toggleButton.style.top = "10px";
        toggleButton.style.right = "10px";
        toggleButton.style.padding = "5px 10px";
        toggleButton.style.backgroundColor = "#1976D2"; // Primary Blue
        toggleButton.style.color = "white";
        toggleButton.style.border = "none";
        toggleButton.style.borderRadius = "3px";
        toggleButton.style.cursor = "pointer";
        toggleButton.style.zIndex = "1000"; // Ensure it appears above the editor

        // Ensure the editor container has relative positioning for absolute button
        editorContainer.style.position = "relative";

        // Append the toggle button to the editor container
        editorContainer.appendChild(toggleButton);

        let isDiffMode = true;

        // Toggle function to switch between Diff Mode and Normal Mode
        toggleButton.addEventListener("click", function () {
          if (isDiffMode) {
            // Switch to Normal Mode
            diffEditorInstance.dispose();

            // Create a new Monaco Editor with the modified content
            const normalEditor = monaco.editor.create(
              document.getElementById(editorContainerId),
              {
                value: newContent,
                language: "markdown",
                readOnly: false,
                automaticLayout: true,
                wordWrap: "on",
              }
            );

            // Store the normal editor instance
            editorContainer.editorInstance = normalEditor;

            // Update toggle state and button text
            isDiffMode = false;
            toggleButton.textContent = "Show Diff";
          } else {
            // Switch back to Diff Mode
            editorContainer.editorInstance.dispose();

            // Recreate the Diff Editor with original and new content
            createMonacoDiffEditor(
              editorContainerId,
              originalContent,
              newContent
            )
              .then(function (newDiffEditor) {
                editorContainer.editorInstance = newDiffEditor;
                isDiffMode = true;
                toggleButton.textContent = "Show Normal";
              })
              .catch(function (error) {
                console.error("Error recreating Diff Editor:", error);
              });
          }
        });
      }
    })
    .catch(function (error) {
      console.error("Error creating Diff Editor:", error);
    });
}

function handleMultipleSolutions(data) {
  try {
    const columnMax = data.column_max;
    const columnIndex = data.column_id; // Index numérique
    const agent_columnId = `${data.agent_name}-${data.column_id}`; // Identifiant chaîne pour les éléments HTML
    const content = data.message;

    let solutionColumnsContainer = document.querySelector(
      `#message-accordion > div[data-agent="${data.agent_name}"] > div > .solution-columns`
    );

    // Si c'est la première solution, on recrée les colonnes
    if (columnIndex === 0 && solutionColumnsContainer) {
      solutionColumnsContainer.remove();
      solutionColumnsContainer = null;
      // Supprimer le bouton "Continue" s'il existe
      let continueButton = document.getElementById(
        `continue-button-${data.agent_name}`
      );
      if (continueButton) {
        continueButton.remove();
      }
      let pagination = document.getElementById("pagination");
      if (pagination) {
        pagination.remove();
      }
    }

    if (!solutionColumnsContainer) {
      solutionColumnsContainer = document.createElement("div");
      solutionColumnsContainer.className = "solution-columns";
      let agentAccordion = document.querySelector(
        `#message-accordion > div[data-agent="${data.agent_name}"] > div`
      );
      if (!agentAccordion) {
        agentAccordion = document.createElement("div");
        agentAccordion.setAttribute("data-agent", data.agent_name);
        agentAccordion.innerHTML = `<h3>${data.agent_name}</h3><div></div>`;
        document
          .querySelector("#message-accordion")
          .appendChild(agentAccordion);
      }

      agentAccordion.appendChild(solutionColumnsContainer);

      for (let i = 0; i < columnMax; i++) {
        const colId = `${data.agent_name}-${i}`;
        const column = document.createElement("div");
        column.className = "solution-column";
        column.id = `column-${colId}`;
        column.innerHTML = `<button class="expand-button"><i class="fas fa-expand"></i></button><h4>Answer ${i}</h4>`;
        solutionColumnsContainer.appendChild(column);

        column.innerHTML += `</div><div class="overlay"></div>`;
      }
    }

    const targetColumn = document.getElementById(`column-${agent_columnId}`);
    const formattedMessage = formatMessage(content);
    if (targetColumn) {
      // Create a unique ID for the editor container
      let editorId = `editor-${agent_columnId}`;
      targetColumn.innerHTML += `<div id="${editorId}" style="height: 600px;"></div>`;
      // Initialize the Monaco Editor
      createMonacoEditor(
        editorId,
        content,
        "markdown",
        data.agent_name,
        data.column_id
      ).then(function (editor) {
        targetColumn.editorInstance = editor;
      });

      const evaluationType = agentEvaluations[data.agent_name] || "default";

      // Add Select and Remove buttons if more than 1 solution
      let selectionButtonsHtml =
        data.column_max > 1
          ? `<div style="text-align: center;"><button id="delete-${agent_columnId}"><i class="fa-regular fa-trash-can"></i>Remove</button></div>`
          : "";
      //`<div style="text-align: center;"><button id="select-${agent_columnId}"><i class="fa-regular fa-check-circle"></i> Select</button></div>
      //<div style="text-align: center;"><button id="delete-${agent_columnId}"><i class="fa-regular fa-trash-can"></i> Remove</button></div>` : "";
      targetColumn.innerHTML += `
            <div style="text-align: center;">
                <div style="display: flex; justify-content: center; align-items: flex-start;">

                    <div style="display: flex; flex-direction: column; align-items: center; margin-right: 10px;">
                        ${selectionButtonsHtml}
                        <button id="save-${agent_columnId}" style="display: none; margin-top: 5px;"><i class="fa-solid fa-rotate"></i>Save your Edit</button>
                    </div>


                    <div id="comment-monaco-editor-${agent_columnId}" style="margin-right: 10px; width: 750px; height: 250px; position: relative;">
                        <button class="expand-button" id="comment-expand-${agent_columnId}">
                            <i class="fas fa-expand"></i>
                        </button>
                    </div>



                    <div style="display: flex; flex-direction: column; gap: 5px;">
                        <button id="suggestions-${agent_columnId}"><i class="fa-regular fa-paper-plane"></i>Get suggestions</button><br>
                        <button id="evaluate-${agent_columnId}"><i class="fa-solid fa-rotate"></i>Regenerate from instructions</button><br>
                        <button id="regenwithmymodifbutton-${agent_columnId}"><i class="fa-solid fa-rotate"></i>Regenerate from this answer's annotations</button>
                        <button id="regenallbutton-${agent_columnId}"><i class="fa-solid fa-rotate"></i>Regenerate from all answers & annotations</button>
                    </div>
                </div>
            </div>
            `;

      if (evaluationType === "score") {
        targetColumn.innerHTML += `
                    <br>
                    <div style="text-align: center;">
                        <button id="score-${agent_columnId}">Waiting score...</button>
                    </div>
                `;

        targetColumn.innerHTML += `
                <br>
                <div style="text-align: center;" id="error-content-${agent_columnId}"/>
            `;
      }

      // Add the menu options for column
      targetColumn.innerHTML += `<div id="menu-options-column-${agent_columnId}"></div>`;

      // Refresh value on value change
      $(`#slider-${agent_columnId}`).on("input", function () {
        // If value is -1 display "Unscored" instead
        if (this.value === "-1") {
          $(`#value-${agent_columnId}`).text("Unscored");
        } else {
          $(`#value-${agent_columnId}`).text(this.value);
        }
      });

      monacoLoaderPromise.then(function () {
        const editorContainer = document.getElementById(
          `comment-monaco-editor-${agent_columnId}`
        );
        // Apply a black border around the container
        editorContainer.style.border = "2px solid silver";
        editorContainer.style.borderRadius = "5px";
        editorContainer.style.textAlign = "left";
        const editor = monaco.editor.create(
          document.getElementById(`comment-monaco-editor-${agent_columnId}`),
          {
            value: "", // Provide default content if needed
            language: "markdown", // Change language if necessary
            automaticLayout: true,
            wordWrap: "on",
            folding: true,
          }
        );
        // Optionally, store the editor instance for further interaction
        window[`editorInstance_${agent_columnId}`] = editor;
      });

      let toggleTimeouts = {}; // To store timeout references by `agent_columnId`

      function enableGenButtons(
        isEnabled,
        agent_columnId = null,
        selectedButton = null,
        timeoutReEnable = 30000
      ) {
        // Determine the button selectors, handling the case when agent_columnId is null
        let buttonSelectors =
          agent_columnId === null
            ? [
                "[id^=evaluate-]",
                "[id^=regenallbutton-]",
                "[id^=regenwithmymodifbutton-]",
                "[id^=suggestions-]",
              ]
            : [
                `#evaluate-${agent_columnId}`,
                `#regenallbutton-${agent_columnId}`,
                `#regenwithmymodifbutton-${agent_columnId}`,
                `#suggestions-${agent_columnId}`,
              ];

        let buttons = buttonSelectors.map((selector) => $(selector));
        let targetButton = $(selectedButton);

        if (isEnabled) {
          // Enable buttons
          buttons.forEach((btn) => btn.prop("disabled", false));

          // If there's an active timeout, clear it
          if (agent_columnId && toggleTimeouts[agent_columnId]) {
            clearTimeout(toggleTimeouts[agent_columnId]);
            delete toggleTimeouts[agent_columnId]; // Remove reference after clearing
          }
        } else {
          // Disable buttons
          buttons.forEach((btn) => btn.prop("disabled", true));

          // If a timeout is provided, set it for enabling buttons after the specified time
          if (timeoutReEnable > 0 && agent_columnId !== null) {
            toggleTimeouts[agent_columnId] = setTimeout(() => {
              buttons.forEach((btn) => btn.prop("disabled", false));
              delete toggleTimeouts[agent_columnId]; // Clean up after timeout completes
            }, timeoutReEnable);
          }
        }
      }

      $(`#save-${agent_columnId}`).click(function () {
        // Retrieve the target column and editor instance
        let targetColumn = document.getElementById(`column-${agent_columnId}`);
        if (!targetColumn || !targetColumn.editorInstance) {
          console.error(
            `Editor instance not found for column-${agent_columnId}.`
          );
          return;
        }

        let editor = targetColumn.editorInstance;
        let currentContent = editor.getValue();

        // Prepare the data for saving
        let requestData = { answer: currentContent };

        // Send the save function call manually
        sendFunctionCall(
          data.agent_name,
          "update_answer",
          requestData,
          function (responseMessage) {
            console.log("Save response:", responseMessage);
            alert(responseMessage);
            document.querySelector(`#save-${agent_columnId}`).style.display =
              "none";
          }
        );
      });

      $(`#suggestions-${agent_columnId}`).click(function () {
        let editor_content = targetColumn.editorInstance.getValue();
        let output_id = data.column_id;
        sendFunctionCall(
          data.agent_name,
          "generate_instructions_feedback",
          { inference_result_content: editor_content, output_id: output_id },
          function (responseMessage) {
            console.log("responseMessage:", responseMessage);
            let suggestions = responseMessage;
            annotationswithID = suggestions.suggestions;
            let comment_editor =
              window[`editorInstance_${currentAgent}-${suggestions.output_id}`];
            comment_editor.setValue(annotationswithID);
          }
        );
      });

      $(`#evaluate-${agent_columnId}`).click(function () {
        clearTimeout(debounceSaveTimer); // to avoid saving at the same time
        // Check if the input field is empty
        let comment_editor = window[`editorInstance_${agent_columnId}`];
        // Add text in the comment editor
        let comment = comment_editor.getValue();
        if (comment === "") {
          alert("Write your instructions first");
          return;
        }

        // Disable the button and update its text
        enableGenButtons(false);

        // Retrieve the score
        let score;
        if ($(`#slider-${agent_columnId}`).length) {
          score = parseFloat($(`#slider-${agent_columnId}`).val()) / 10;
        } else {
          score = $(`#compliance-${agent_columnId}`).val();
        }

        // Retrieve the target column and editor instance
        let targetColumn = document.getElementById(`column-${agent_columnId}`);
        if (!targetColumn || !targetColumn.editorInstance) {
          console.error(
            `Editor instance not found for column-${agent_columnId}.`
          );
          alert("Associated editor not found.");
          enableGenButtons(true);
          return;
        }

        let editor = targetColumn.editorInstance;
        let originalContent = editor.getValue();

        // Prepare data for sendFunctionCall
        let requestData = {
          suggestions: comment,
          text_content: originalContent,
          text_has_annotations: false,
        };

        // Send the function call
        sendFunctionCall(
          data.agent_name,
          "critic_answer",
          requestData,
          function (responseMessage) {
            console.log("responseMessage:", responseMessage);

            // Update the Monaco Editor content with the formatted response
            let formattedResponse = formatMessage(responseMessage);
            editor.setValue(formattedResponse);
            //activateCodeComparison(agent_columnId, requestData.originalContent, formattedResponse);

            // Re-enable the button and reset its text
            enableGenButtons(true);

            // Log the action
            let actionDescription = `User evaluated solution ${agent_columnId} with score ${score} and comment "${comment}"`;
            addHistoryEntry(actionDescription);
          }
        );
      });

      $(`#regenallbutton-${agent_columnId}`).click(function () {
        clearTimeout(debounceSaveTimer); // to avoid saving at the same time
        // Disable all generation buttons
        enableGenButtons(false);

        // Extract agent name from agent_columnId (assuming format 'agentName-columnId')
        let [agentName, columnId] = agent_columnId.split("-");

        // Select all editors associated with this agent
        let allColumns = document.querySelectorAll(
          `.solution-column[id^='column-${agentName}-']`
        );
        let annotatedTexts = [];

        allColumns.forEach(function (column) {
          let editor = column.editorInstance;
          if (editor) {
            let decorations = editor.getModel().getAllDecorations();
            let hasAnnotations = decorations.some(
              (decoration) =>
                decoration.options.inlineClassName === "applied-annotation"
            );
            if (hasAnnotations) {
              let text = editor.getValue();
              console.log("Annotated text:", text);
              annotatedTexts.push(text);
            }
          }
        });

        if (annotatedTexts.length === 0) {
          alert("Annotate some of the answers first to be applied");
          enableGenButtons(true);
          return;
        }

        // Prepare data for sendFunctionCall
        let requestData = {
          suggestions: "",
          text_content: annotatedTexts,
          annotation_format: $("#annotation-type").val(),
        };

        // Send the function call
        sendFunctionCall(
          agentName,
          "critic_answer",
          requestData,
          function (responseMessage) {
            console.log("responseMessage:", responseMessage);

            // Format the response appropriately
            let formattedResponse = formatMessage(responseMessage);

            // Update only the target Monaco Editor (assuming you want to update the first one or a specific one)
            // Modify this part based on your specific logic for selecting which editor to update
            let targetEditor = allColumns[columnId].editorInstance; // Example: updating the first editor
            if (targetEditor) {
              targetEditor.setValue(formattedResponse);
            }

            // Re-enable all generation buttons
            enableGenButtons(true);

            // Log the action
            let actionDescription = `User requested regeneration of solutions for ${agentName}`;
            addHistoryEntry(actionDescription);
          }
        );
      });
      $(`#regenwithmymodifbutton-${agent_columnId}`).click(function () {
        clearTimeout(debounceSaveTimer); // to avoid saving at the same time
        console.log("regenwithmymodifbutton clicked");
        let targetColumn = document.getElementById(`column-${agent_columnId}`);

        if (!targetColumn || !targetColumn.editorInstance) {
          console.error(
            `Editor instance not found for column-${agent_columnId}.`
          );
          alert("Associated editor not found.");
          return;
        }

        let editor = targetColumn.editorInstance;

        // Check if at least one annotation exists
        let model = editor.getModel();
        let allDecorations = model.getAllDecorations();
        let hasAnnotations = allDecorations.some(
          (decoration) =>
            decoration.options.inlineClassName === "applied-annotation"
        );

        if (!hasAnnotations) {
          alert("Annotate some of the answers first to be applied");
          return; // Block the action if no annotations are present
        }

        // Retrieve and process the editor content
        let text = editor.getValue();
        let textWithoutHtml = text.replace(/<[^>]*>/g, ""); // TODO: I removed and don't understand the purpose

        // Disable the button and update its text
        enableGenButtons(false);

        // Prepare data for sendFunctionCall
        let requestData = {
          suggestions: "",
          text_content: text, // TODO: textWithoutHtml IGNORED => check !
          annotation_format: $("#annotation-type").val(),
        };

        // Send the function call
        sendFunctionCall(
          data.agent_name,
          "critic_answer",
          requestData,
          function (responseMessage) {
            console.log("responseMessage:", responseMessage);

            // Update the Monaco Editor content with the formatted response
            let formattedResponse = formatMessage(responseMessage);
            editor.setValue(formattedResponse);

            // Re-enable the button and reset its text
            enableGenButtons(true);

            // Log the action
            let actionDescription = `User requested regeneration of solution ${agent_columnId} with modifications`;
            addHistoryEntry(actionDescription);
          }
        );
      });

      $(`#delete-${agent_columnId}`).click(function () {
        let targetColumn = document.getElementById(`column-${agent_columnId}`);
        let isMarkedForDeletion = targetColumn.classList.contains("deleted");
        console.log(`#delete: column-${agent_columnId}`);
        // Toggle the 'deleted' class
        if (isMarkedForDeletion) {
          console.log("Removing deleted class");
          targetColumn.classList.remove("deleted");
          targetColumn.style.backgroundColor = "";
        } else {
          console.log("Add deleted class");
          targetColumn.classList.add("deleted");
          targetColumn.style.backgroundColor = "lightcoral";
        }

        // Recompute columnsToKeep
        let columnsToKeep = getColumnsToKeep(data.agent_name); // currentAgent
        // Send the updated list to the backend
        console.log(data.agent_name + " - Columns to keep:", columnsToKeep);
        sendFunctionCall(data.agent_name, "set_selected_outputs", {
          selected_outputs: columnsToKeep,
        });
      });
    }

    // Gérer la pagination et le bouton "Continue"
    if (columnIndex === columnMax - 1 && columnMax > 4) {
      // Code de pagination
    }

    if (columnIndex === columnMax - 1) {
      // Si le bouton existe déjà, ne pas le recréer
      if (document.getElementById(`continue-button-${data.agent_name}`)) {
        return;
      }
      // Ajouter le bouton "Continue"
      if (
        data.agent_name === "CodingAgent" ||
        data.agent_name === "PlannerAgent"
      ) {
        let $continueButton = $(
          `<button id="continue-button-${data.agent_name}" disabled>Keep 0, 1 or more propositions, then HIT continue here</button>`
        );
        $(solutionColumnsContainer).after($continueButton);
        $continueButton.click(function () {
          console.log("Continue button clicked");
          let columnsToKeep = getColumnsToKeep(data.agent_name);
          console.log("Columns to keep:", columnsToKeep.join(","));
          sendWebSocketMessage({ message: columnsToKeep.join(",") });
          number_validated_functions += columnsToKeep.length;
          document.getElementById("number-validated-functions").textContent =
            number_validated_functions;
          let actionDescription = `User clicked continue, keeping solutions: ${columnsToKeep
            .map((num) => num + 1)
            .join(", ")}`;
          addHistoryEntry(actionDescription);
        });
      }
    }

    if (columnMax > 4) {
      let $columns = $(`.solution-column[id^='column-${data.agent_name}-']`);
      $columns.hide();
      $columns.slice(0, 4).show();
    }
  } catch (error) {
    console.error("Error handling multiple solutions:", error);
  }
}

function adjustColumnWidths() {
  console.log(
    "Adjusting column widths with",
    document.querySelectorAll(".solution-columns .solution-column").length,
    "columns"
  );
  const columns = document.querySelectorAll(
    ".solution-columns .solution-column"
  );
  const numColumns = columns.length;
  columns.forEach((column) => {
    column.style.width = `calc(100% / ${numColumns})`;
  });
}

function handleEvaluationResults(data) {
  let $agentAccordion = $(
    `#message-accordion > div[data-agent="${data.agent_name}"]`
  );
  if ($agentAccordion.length === 0) {
    $agentAccordion = $(
      `<div data-agent="${data.agent_name}"><h3>${data.agent_name}</h3><div></div></div>`
    );
    $("#message-accordion").append($agentAccordion);
  }

  // Process evaluation results based on agent's evaluation type
  const evaluationType = agentEvaluations[data.agent_name] || "default";
  console.log(evaluationType, data);

  if (evaluationType === "score") {
    // Handle scoring results
    // Similar to your existing code for scores
    // Update score buttons or display score information
    // You can adapt this part based on how you receive and display scores
    // COPIED CODE
    // Enable "continue" button
    $(`#continue-button-${data.agent_name}`).prop("disabled", false);
    let $scoreButtons = $agentAccordion.find("button[id^='score-']");
    $scoreButtons.each(function (index, button) {
      if (data.column_id !== index) {
        return;
      }

      // Regex patterns
      // const indexRegex = /^(\d+)\./;
      const statusRegex = /\[\d+m(SUCCESS|FAILED)\[\d+m/;
      const scoreRegex = /SCORE:\s+(\[.*?\])/;
      const timeRegex = /TIME:\s+([\d.]+)s/;
      const codeRegex = /CODE:\s+(\[.*\])$/;

      // Ensure data and message are valid
      if (!data || !data.message) {
        console.error("Invalid data or message");
        return;
      }

      // Extract values
      // const indexMatch = data.match(indexRegex);
      const statusMatch = data.message.match(statusRegex);
      const scoreMatch = data.message.match(scoreRegex);
      const timeMatch = data.message.match(timeRegex);
      const codeMatch = data.message.match(codeRegex);

      // Process the message
      if (statusMatch) {
        console.log("Match found:", statusMatch);
        const score = statusMatch[1];
        button.textContent = score;
        if (score === "SUCCESS") {
          button.style.backgroundColor = "green";
        } else {
          button.style.backgroundColor = "red";
        }

        if (scoreMatch) {
          const fullText = scoreMatch[1].trim();
          const timeT = parseFloat(timeMatch[1]).toFixed(1); // Convert time string to a float and fix precision. // MODIFIED
          button.textContent +=
            "\n" + fullText.substring(0, 70) + "...[see DETAILS]";

          const detailScores = formatPythonDict2Table(codeMatch[1]);

          // Remove any previously attached listener
          if (button._clickListener) {
            button.removeEventListener("click", button._clickListener);
          }

          // Define the new listener and store its reference
          const buttonClick = () => {
            alert(
              `Test score of solution ${index} in ${timeT}s:\n${detailScores}`
            );
          };
          button._clickListener = buttonClick;

          // Add the new listener
          button.addEventListener("click", buttonClick);
        }
      } else {
        button.textContent = "Scoring failed";
        button.style.backgroundColor = "red";
      }
    });
    // END COPY
  } else if (evaluationType === "compliance") {
    // Handle compliance results
    // Update compliance status or display compliance information
  } else {
    // Handle default evaluation results
    // Display evaluation results in a general way
  }
}

function formatMarkdown(message) {
  return marked.parse(message);
}

// Listener for the task selection dropdown on click and not on change
$("#task-selection").on("click", function () {
  updateTaskList("None");
});

// Listener for the task selection dropdown on change
$("#task-selection").on("change", function () {
  let selectedTaskId = $(this).val();
  let selectedTaskText = $(this).find("option:selected").text();
  const modal = document.createElement("div");
  modal.id = "task-modal";
  modal.style.position = "fixed";
  modal.style.top = "50%";
  modal.style.left = "50%";
  modal.style.transform = "translate(-50%, -50%)";
  modal.style.backgroundColor = "white";
  modal.style.padding = "20px";
  modal.style.borderRadius = "10px";
  modal.style.boxShadow = "0px 4px 6px rgba(0, 0, 0, 0.1)";
  modal.style.zIndex = "1000";
  modal.style.width = "80%";

  modal.innerHTML =
    `
                <h3 style="text-align: center;">Task Configuration</h3>
                <p style="text-align: center; margin-bottom: 20px;">` +
    selectedTaskText +
    `</p>
                <p style="text-align: center; margin-bottom: 20px;">You are configuring settings for the selected task. Customize the automatic mode for each agent below.</p>
                <h3 style="text-align: center;">Automatic Configuration</h3>
                <div style="text-align: center; margin-bottom: 15px;"><input type="text" id="task-details" placeholder="ENTER info, next user or action on the collaboration..." style="width: 80%; padding: 10px; border-radius: 5px; border: 1px solid #ccc;"></div>
                <style>#task-details::placeholder{color: red;}</style>
                <div style="display: flex; justify-content: space-between; gap: 20px;">
                    ${[
                      "TaskIdentificationAgent",
                      "CodingAgent",
                      "ValidationAgent",
                      "CapitalizationAgent",
                    ]
                      .map((agent, index) => {
                        const agentId = index + 1;
                        const isCodingAgent = agent === "CodingAgent";
                        return `
                        <div style="flex: 1; border: 1px solid #ccc; border-radius: 10px; padding: 10px; text-align: center;">

                            <label style="font-weight: bold;">${agent}</label>

                            <div style="display: inline-flex; align-items: center; gap: 10px;">
                                <label>Skip rounds:</label>
                                <input type="number" id="int-input-agent${agentId}" value="0" placeholder="Enter value" style="width: 50px;">
                                <label><input type="checkbox" id="full-auto-agent${agentId}"> Full auto</label>
                            </div>
                            <div>
                                <label><input type="checkbox" id="recommendations-agent${agentId}" disabled> Automatic application of recommendations</label>
                            </div>
                            ${
                              isCodingAgent
                                ? `
                            <div>
                                <label>Max autofix:</label>
                                <select id="max-autofix-agent${agentId}" disabled>
                                    <option value="0">0</option>
                                    <option value="1">1</option>
                                    <option value="2">2</option>
                                </select>
                            </div>
                            `
                                : ""
                            }
                            <div>
                                <label>LLM:</label>
                                <select id="selectlist-agent${agentId}">
                                    <option value="default_llm">Default LLM</option>
                                    <option value="premium_llm">Premium LLM</option>
                                    <option value="3_majority_chain">3_majority_chain</option>
                                    <option value="10_majority_chain">10_majority_chain</option>
                                </select>
                            </div>
                            <div>
                                <label>Number of inferences:</label>
                                <select id="select-value-agent${agentId}">
                                    <option value="0" selected disabled>0</option>
                                    <option value="1">1</option>
                                    <option value="2">2</option>
                                    <option value="3">3</option>
                                    <option value="4">4</option>
                                </select>
                            </div>
                            <div>
                                <label>LLM Temperature: <span id="slider-value-agent${agentId}">0.5</span></label>
                                <input type="range" id="slider-agent${agentId}" min="0" max="1" step="0.1" value="0.5">
                            </div>
                        </div>
                        `;
                      })
                      .join("")}
                </div>
                <div style="margin-top: 20px; text-align: right;">
                    <button id="goto-task-button">Go to Task</button>
                    <button id="cancel-button">Cancel</button>
                </div>
            `;

  document.body.appendChild(modal);

  const overlay = document.createElement("div");
  overlay.id = "modal-overlay";
  overlay.style.position = "fixed";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.width = "100%";
  overlay.style.height = "100%";
  overlay.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
  overlay.style.zIndex = "999";

  document.body.appendChild(overlay);

  overlay.addEventListener("click", function () {
    modal.remove();
    overlay.remove();
  });

  ["1", "2", "3", "4"].forEach((agent) => {
    const checkbox = document.getElementById(`checkbox-agent${agent}`);
    const input = document.getElementById(`int-input-agent${agent}`);
    const fullAuto = document.getElementById(`full-auto-agent${agent}`);
    const recommendations = document.getElementById(
      `recommendations-agent${agent}`
    );
    const select = document.getElementById(`selectlist-agent${agent}`);
    const selectValue = document.getElementById(`select-value-agent${agent}`);
    const slider = document.getElementById(`slider-agent${agent}`);
    const sliderValue = document.getElementById(`slider-value-agent${agent}`);
    const autofix =
      agent === "2"
        ? document.getElementById(`max-autofix-agent${agent}`)
        : null;

    input.addEventListener("input", function () {
      const hasValue = parseInt(this.value, 10) > 0;
      recommendations.disabled = !hasValue;
      if (autofix) autofix.disabled = !hasValue;
    });

    fullAuto.addEventListener("change", function () {
      input.value = this.checked ? 999 : 0;
      const hasValue = parseInt(input.value, 10) > 0;
      recommendations.disabled = !hasValue;
      if (autofix) autofix.disabled = !hasValue;
    });

    slider.addEventListener("input", function () {
      sliderValue.textContent = slider.value;
    });
  });

  document
    .getElementById("goto-task-button")
    .addEventListener("click", function () {
      let is_automatic = false;
      console.log("Gototask button clicked !");
      const specialCriteria = {};
      ["1", "2", "3", "4"].forEach((agent, index) => {
        const agentName = [
          "TaskIdentificationAgent",
          "CodingAgent",
          "ValidationAgent",
          "CapitalizationAgent",
        ][index];

        const autoNRounds =
          document.getElementById(`int-input-agent${agent}`).value || 0;
        const defaultLLMChoice = document.getElementById(
          `selectlist-agent${agent}`
        ).value;
        const recommendCritics = document.getElementById(
          `recommendations-agent${agent}`
        ).checked;
        const maxAutofix =
          agentName === "CodingAgent"
            ? document.getElementById(`max-autofix-agent${agent}`).value || 0
            : 0;
        const numParallelInferences =
          document.getElementById(`select-value-agent${agent}`).value || 1;
        const temperatureMax =
          document.getElementById(`slider-agent${agent}`).value || 0.5;

        // Placeholder for replace_if_exists_function, you can add specific logic as needed
        const replaceIfExistsFunction = false;

        if (parseInt(autoNRounds, 10) > 0) {
          is_automatic = true;
        }

        specialCriteria[`${agentName}#auto_n_rounds`] = parseInt(
          autoNRounds,
          10
        );
        specialCriteria[`${agentName}#default_llm_choice`] = defaultLLMChoice;
        specialCriteria[`${agentName}#recommend_critics`] = recommendCritics;
        specialCriteria[`${agentName}#max_autofix`] = parseInt(maxAutofix, 10);
        specialCriteria[`${agentName}#num_parallel_inferences`] = parseInt(
          numParallelInferences,
          10
        );
        specialCriteria[`${agentName}#temperature_max`] =
          parseFloat(temperatureMax);
        specialCriteria[`${agentName}#replace_if_exists_function`] =
          replaceIfExistsFunction;
      });
      console.log("Criteria : " + specialCriteria);
      let automatic = "None";
      if (is_automatic) {
        automatic = "True";
      }
      let taskDetails = document.getElementById("task-details").value.trim();
      sendFunctionCall(
        currentAgent,
        "goto_task",
        [selectedTaskId, automatic, specialCriteria, taskDetails],
        function (response) {
          const url = response;
          console.log("Link to goto", url);
          // Open the URL in a new tab
          // window.open(url, '_blank');
          alert("Task available here : " + url);
        }
      );
      modal.remove();
      overlay.remove();
    });

  document
    .getElementById("cancel-button")
    .addEventListener("click", function () {
      modal.remove();
      overlay.remove();
    });
});

function updateTaskList(id) {
  if (id === "None") {
    console.log("Getting task list");
    sendFunctionCall(
      currentAgent,
      "get_tasks",
      { id_last_task: "None" },
      function (response) {
        let tasks = JSON.parse(response);
        console.log("Task list:", tasks);
        // Référence à l'élément <select>
        const selectElement = document.getElementById("task-selection");

        // Vider la liste actuelle
        selectElement.innerHTML = "";

        // Ajouter chaque tâche au <select>
        tasks.forEach((task) => {
          const option = document.createElement("option");
          option.value = task.task_id; // On peut utiliser l'ID comme valeur
          let username = task.user_id;
          if (task.user_id === uniqueId.toString().toUpperCase()) {
            username = "You";
          }
          let details = task.task_details ? ` - ${task.task_details}` : "";
          option.textContent = `${task.task_id} - ${task.date.substring(
            0,
            16
          )} - ${task.agent_name} - ${task.function_name} - ${
            task.before_after.charAt(0).toUpperCase() +
            task.before_after.slice(1)
          } - ${username}${details}`;
          selectElement.appendChild(option);
        });
        filterTaskList();
      }
    );
  } else {
    console.log("Getting task list");
    sendFunctionCall(
      currentAgent,
      "get_tasks",
      { id_last_task: id },
      function (response) {
        let tasks = JSON.parse(response);
        console.log("Task list:", tasks);
        // Référence à l'élément <select>
        const selectElement = document.getElementById("task-selection");

        // Vider la liste actuelle
        selectElement.innerHTML = "";

        // Ajouter chaque tâche au <select>
        tasks.forEach((task) => {
          const option = document.createElement("option");
          option.value = task.task_id; // On peut utiliser l'ID comme valeur
          let username = task.user_id;
          if (task.user_id === uniqueId.toString().toUpperCase()) {
            username = "You";
          }
          let details = task.task_details ? ` - ${task.task_details}` : "";
          option.textContent = `${task.date.substring(0, 16)} - ${
            task.agent_name
          } - ${username}${details}`; // Texte visible by user
          selectElement.appendChild(option);
        });
        filterTaskList();
      }
    );
  }
}

// Adds the listener for the expand button of the solution columns
$(document).on("click", ".expand-button", function () {
  // If the expand button has an id
  if (this.id) {
    editorContainer = document.getElementById(
      this.id.replace("expand-", "monaco-editor-")
    );
    console.log(this.id.replace("expand-", "editor-"));
    expandButton = this;
    console.log("Expanding editor");
    const isExpanded = editorContainer.classList.contains("expanded");
    if (!isExpanded) {
      editorContainer.classList.add("expanded");
      // Agrandir l'éditeur
      editorContainer.style.width = "100%";
      editorContainer.style.height = "400px";
      editorContainer.style.position = "absolute";
      editorContainer.style.top = "400px";
      editorContainer.style.left = "0";
      editorContainer.style.zIndex = "1000";
      expandButton.innerHTML = `<i class="fas fa-compress"></i>`;
    } else {
      editorContainer.classList.remove("expanded");
      // Réduire l'éditeur
      editorContainer.style.width = "750px";
      editorContainer.style.height = "250px";
      editorContainer.style.position = "relative";
      editorContainer.style.top = "";
      editorContainer.style.left = "";
      editorContainer.style.zIndex = "";
      expandButton.innerHTML = `<i class="fas fa-expand"></i>`;
      // Resize the width of a solution column
      adjustColumnWidths();
    }
  } else {
    let $column = $(this).closest(".solution-column");
    let $icon = $(this).find("i");
    if ($column.hasClass("expanded")) {
      $column.removeClass("expanded");
      // Modify the icon to show the expand icon
      $icon.removeClass("fa-compress").addClass("fa-expand");
      adjustColumnWidths();
    } else {
      $column.addClass("expanded");
      // Modify the icon to show the compress icon
      $icon.removeClass("fa-expand").addClass("fa-compress");
    }
  }
});

function scrollToBottom() {
  let mainContent = document.getElementById("main-content");
  mainContent.scrollTop = mainContent.scrollHeight;
}

$("#auto-open-accordion").change(function () {
  if ($(this).is(":checked")) {
    scrollToBottom();
  }
  // Update the checkbox at the bottom of the page
  $("#auto-open-accordion-bottom").prop("checked", $(this).is(":checked"));
});

$("#auto-open-accordion-bottom").change(function () {
  if ($(this).is(":checked")) {
    scrollToBottom();
  }
  // Update the checkbox at the top of the page
  $("#auto-open-accordion").prop("checked", $(this).is(":checked"));
});

function formatMessage(message, raw = true) {
  if (raw) return message;
  // Remove ANSI color codes
  let cleanString = message.replace(/\u001b\[\d+m/g, "");
  // Remove all ANSI escape sequences
  cleanString = cleanString.replace(/\u\d+b/g, "");

  // Regular expression to match code blocks
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;

  // Extract code blocks and replace with placeholders
  let codeBlocks = [];
  cleanString = cleanString.replace(
    codeBlockRegex,
    function (match, lang, code) {
      codeBlocks.push({ lang: lang, code: code });
      return `[[CODE_BLOCK_${codeBlocks.length - 1}]]`;
    }
  );

  // Escape HTML characters in the rest of the text
  cleanString = escapeHtml(cleanString);

  // Replace remaining line breaks in the rest of the text with <br>
  cleanString = cleanString.replace(/\n/g, "<br>");

  // Replace placeholders with highlighted code
  codeBlocks.forEach(function (block, index) {
    let highlightedCode;
    if (block.lang && hljs.getLanguage(block.lang)) {
      highlightedCode = hljs.highlight(block.lang, block.code).value;
    } else {
      highlightedCode = hljs.highlightAuto(block.code).value;
    }
    // Preserve line breaks and indentation in code blocks
    cleanString = cleanString.replace(
      `[[CODE_BLOCK_${index}]]`,
      `<pre><code>${highlightedCode}</code></pre>`
    );
  });

  return `${cleanString}`;
}

function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
    "`": "&#096;",
    "/": "&#x2F;",
  };
  return text.replace(/[&<>"'`/]/g, function (m) {
    return map[m];
  });
}

function isPythonCode(message) {
  // Check if message contains code blocks with python code
  return /```python[\s\S]*?```/i.test(message);
}

function isPromptMessage(message) {
  // Check if the message looks like a prompt
  return /^Reasoning\b|^You are\b|^As an AI language model\b|^Please\b|^Write\b|^Generate\b|^Explain\b/i.test(
    message
  );
}

$(document).on("click", ".collapsible-header", function () {
  $(this).parent(".collapsible-message").toggleClass("open");
});

function displayInputField() {
  $(".input-area").show();
  $("#user-input").show();
  $("#send-button").show();
  if (isInputRequired) {
    $("#user-input").prop("required", true);
    $("#user-input").css("border", "2px solid red");
  } else {
    $("#user-input").prop("required", false);
    $("#user-input").css("border", "");
  }

  // Hide buffering
  console.log("Hiding buffering");
  $(".buffering-circle").hide();
}

function displayMenuOptions(data) {
  let agentAccordion = document.querySelector(
    `#message-accordion > div[data-agent="${data.agent_name}"] > div`
  );
  if (!agentAccordion) {
    // Create agent accordion if it doesn't exist
    agentAccordion = document.createElement("div");
    agentAccordion.setAttribute("data-agent", data.agent_name);
    agentAccordion.innerHTML = `<h3>${data.agent_name}</h3><div></div>`;
    document.querySelector("#message-accordion").appendChild(agentAccordion);
  }
  let message = data.message;

  // Adds a message on top of the menu options depending on the message type
  let infos = "";
  let $menuMessage;
  if (message.includes("AFTER")) {
    infos = `AFTER ${data.agent_name}`;
    $menuMessage = $(``);
  } else if (message.includes("BEFORE")) {
    infos = `BEFORE ${data.agent_name} inference MENU : Choose an action below`;
    $menuMessage = $(
      `<div style="font-weight: bold; font-size: 24px; color: #1976D2;">${infos}</div>`
    );
  } else {
    infos = "Choose an action below";
    $menuMessage = $(
      `<div style="font-weight: bold; font-size: 24px;">${infos}</div>`
    );
  }

  if (data.column_id !== undefined && data.column_max > 1) {
    console.log(
      `Should display menu in columns here  - column_id:${data.column_id} max:${data.column_max}`
    );
    console.log("Displaying menu options in a separate column");
    let columnIdentifier = `${data.agent_name}-${data.column_id}`;
    // retrieve the column
    let $menuOptionsColumn = $(`#menu-options-column-${columnIdentifier}`);
    // clear the menu options column
    $menuOptionsColumn.empty();
    // Adds infos
    $menuOptionsColumn.append($menuMessage);
    // Existing logic for normal menu options
    let options = message.split("\n").filter((line) => /^\[[A-Z]\]/.test(line));
    options.forEach((option) => {
      let match = option.match(/\[(.)\](.*)/);
      if (match && match[1]) {
        let letter = match[1].toUpperCase();
        if (
          !match[2].includes("Change premium") &&
          !match[2].includes("Exit") &&
          !match[2].includes("Log comments")
        ) {
          if (match[2].includes("Change num of parallel inferences")) {
            // Remove ANSI codes
            let cleanString = match[2].replace(/\u001b\[\d+m/g, "");
            let $option = $(
              `<div class="menu-option" data-option="${letter}">${cleanString}</div>`
            );
            $menuOptionsColumn.append($option);
          } else if (match[2].includes("Continue")) {
            // Set button color to green
            let $option = $(
              `<div class="menu-option" data-option="${letter}" style="background-color: green;">${match[2]}</div>`
            );
            $menuOptionsColumn.append($option);
          } else if (match[2].includes("Change default agent")) {
            // Create a select element with the LLM "default" and "premium"
            let $select = $(
              `<select id="llm-select" name="llm-select" class="menu-option" data-option="${letter}"></select>`
            );
            // Set the default option with the current LLM used and set the text of the default option to this value and get value depends on the text
            let currentLLM = current_llm_in_use;
            if (currentLLM === "premium_llm") {
              $select.append(`<option value="0">default_llm</option>`);
              $select.append(`<option value="1" selected>premium_llm</option>`);
              $select.append(`<option value="2">3_majority_chain</option>`);
              $select.append(`<option value="3">10_majority_chain</option>`);
            } else if (currentLLM === "default_llm") {
              $select.append(`<option value="0" selected>default_llm</option>`);
              $select.append(`<option value="1">premium_llm</option>`);
              $select.append(`<option value="2">3_majority_chain</option>`);
              $select.append(`<option value="3">10_majority_chain</option>`);
            } else if (currentLLM === "3_majority_chain") {
              $select.append(`<option value="0">default_llm</option>`);
              $select.append(`<option value="1">premium_llm</option>`);
              $select.append(
                `<option value="2" selected>3_majority_chain</option>`
              );
              $select.append(`<option value="3">10_majority_chain</option>`);
            } else if (currentLLM === "10_majority_chain") {
              $select.append(`<option value="0">default_llm</option>`);
              $select.append(`<option value="1">premium_llm</option>`);
              $select.append(`<option value="2">3_majority_chain</option>`);
              $select.append(
                `<option value="3" selected>10_majority_chain</option>`
              );
            }

            // Adds the select element to the menu options column
            $menuOptionsColumn.append($select);
          } else {
            // add a specific on click event for the go back button which will clear by doing $(`#message-accordion > div[data-agent="${data.agent_name}"]`).children("div").empty();
            let actionOnclik = "";
            if (match[2].includes("Go back")) {
              actionOnclik = `onclick="console.log(666); let $agentAccordion = $('#message-accordion > div[data-agent=\\'${data.agent_name}\\']'); $agentAccordion.children('div').empty();"`;
              console.log("actionOnclik:" + actionOnclik);
            }
            let $option = $(
              `<div class="menu-option" data-option="${letter}" ${actionOnclik}>${match[2]}</div>`
            );
            console.log("$option:" + $option);
            $menuOptionsColumn.append($option);
          }
        }
      }
    });
  } else {
    console.log("Displaying menu options");
    $("#menu-options").empty();
    // Adds infos
    $("#menu-options").append($menuMessage);
    // Existing logic for normal menu options
    let options = message.split("\n").filter((line) => /^\[[A-Z]\]/.test(line));
    options.forEach((option) => {
      let match = option.match(/\[(.)\](.*)/);
      if (match && match[1]) {
        let letter = match[1].toUpperCase();
        if (
          !match[2].includes("Change premium") &&
          !match[2].includes("Exit") &&
          !match[2].includes("Log comments")
        ) {
          if (match[2].includes("Change num of parallel inferences")) {
            // Remove ANSI codes
            let cleanString = match[2].replace(/\u001b\[\d+m/g, "");
            let $option = $(
              `<div class="menu-option" data-option="${letter}">${cleanString}</div>`
            );
            $("#menu-options").append($option);
          } else if (match[2].includes("Continue")) {
            // Set button color to green
            let $option = $(
              `<div class="menu-option" data-option="${letter}" style="background-color: green;">${match[2]}</div>`
            );
            $("#menu-options").append($option);
          } else if (match[2].includes("Change default agent")) {
            // Create a select element with the LLM "default" and "premium"
            let $select = $(
              `<select id="llm-select" name="llm-select" class="menu-option" data-option="${letter}"></select>`
            );
            // Set the default option with the current LLM used and set the text of the default option to this value and get value depends on the text
            let currentLLM = current_llm_in_use;
            console.log("currentLLM:" + currentLLM);
            if (currentLLM === "premium_llm") {
              $select.append(`<option value="0">default_llm</option>`);
              $select.append(`<option value="1" selected>premium_llm</option>`);
              $select.append(`<option value="2">3_majority_chain</option>`);
              $select.append(`<option value="3">10_majority_chain</option>`);
            } else if (currentLLM === "default_llm") {
              $select.append(`<option value="0" selected>default_llm</option>`);
              $select.append(`<option value="1">premium_llm</option>`);
              $select.append(`<option value="2">3_majority_chain</option>`);
              $select.append(`<option value="3">10_majority_chain</option>`);
            } else if (currentLLM === "3_majority_chain") {
              $select.append(`<option value="0">default_llm</option>`);
              $select.append(`<option value="1">premium_llm</option>`);
              $select.append(
                `<option value="2" selected>3_majority_chain</option>`
              );
              $select.append(`<option value="3">10_majority_chain</option>`);
            } else if (currentLLM === "10_majority_chain") {
              $select.append(`<option value="0">default_llm</option>`);
              $select.append(`<option value="1">premium_llm</option>`);
              $select.append(`<option value="2">3_majority_chain</option>`);
              $select.append(
                `<option value="3" selected>10_majority_chain</option>`
              );
            }

            // Adds the select element to the menu options column
            $("#menu-options").append($select);
          } else {
            let $option = $(
              `<div class="menu-option" data-option="${letter}">${match[2]}</div>`
            );
            $("#menu-options").append($option);
          }
        }
      }
    });
  }
}

function updateFilters(data) {
  if (
    !$("#filter-message-type option[value='" + data.message_type + "']").length
  ) {
    $("#filter-message-type, #filter-message-type-bottom").append(
      new Option(data.message_type, data.message_type)
    );
  }
  if (!$("#filter-agent-type option[value='" + data.agent_name + "']").length) {
    $("#filter-agent-type, #filter-agent-type-bottom").append(
      new Option(data.agent_name, data.agent_name)
    );
  }
}

$("#message-accordion").accordion({
  header: "> div > h3",
  collapsible: true,
  active: false,
  heightStyle: "content",
});

$("#history-accordion").accordion({
  collapsible: true,
  active: false,
  heightStyle: "content",
});

$("#stats-accordion").accordion({
  collapsible: true,
  active: false,
  heightStyle: "content",
});

$("#func-names-accordion").accordion({
  collapsible: true,
  active: false,
  heightStyle: "content",
});

$("#send-button").click(function () {
  sendMessage();
  // Hide options menu
  $("#menu-options").empty();
});

// Add this new event listener for the Enter key
$("#user-input").keypress(function (e) {
  if (e.which == 13) {
    // 13 is the Enter key code
    e.preventDefault(); // Prevent default Enter key behavior
    sendMessage();
    $("#menu-options").empty();
  }
});

// Create a new function to handle sending messages
function sendMessage() {
  let message = $("#user-input").val();
  if (isInputRequired && !message.trim()) {
    return;
  }
  if (!isInputRequired && !message.trim()) {
    message = "Z";
  }
  if (message) {
    sendWebSocketMessage({ message: message });
    $("#user-input").val("");
    inputAwaited = false; // Reset input flag
    isInputRequired = false;
    hideInputField(); // Hide input area after sending message
  }
}

$("#apply-settings").click(function () {
  let settings = {
    defaultLLM: $("#default-llm").val(),
    premiumLLM: $("#premium-llm").val(),
    temperature: $("#temperature").val(),
  };
  sendFunctionCall(currentAgent, "set_default_llmORchain", [
    settings["defaultLLM"],
    settings["temperature"],
  ]);
  sendFunctionCall(currentAgent, "set_premium_llmORchain", [
    settings["premiumLLM"],
    settings["temperature"],
  ]);
  console.log("Settings applied:", settings);
  let actionDescription = `User applied settings: default LLM=${settings.defaultLLM}, premium LLM=${settings.premiumLLM}, temperature=${settings.temperature}`;
  addHistoryEntry(actionDescription);
});

function saveSelection() {
  if (window.getSelection) {
    savedSelection = window.getSelection().getRangeAt(0);
  } else if (document.selection && document.selection.createRange) {
    savedSelection = document.selection.createRange();
  }
}

function restoreSelection() {
  if (savedSelection) {
    if (window.getSelection) {
      let sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(savedSelection);
    } else if (document.selection && savedSelection.select) {
      savedSelection.select();
    }
  }
}

function applyAnnotation(annotationAction, annotationText, selectedText) {
  let annotatedText = `\\${annotationAction}[${annotationText}]{${selectedText}}`;
  //let annotatedText = `\\${annotationAction}{${selectedText}}[${annotationText}]`;
  let boldAnnotatedText = `<strong>${annotatedText}</strong>`;

  restoreSelection();
  let selection = window.getSelection();
  if (!selection.rangeCount) return;

  let range = selection.getRangeAt(0);
  range.deleteContents();

  let newElement = document.createElement("span");
  newElement.innerHTML = boldAnnotatedText;
  newElement.classList.add("applied-annotation"); // to track annotated text
  range.insertNode(newElement);

  // Update selection to include new annotated text
  range.setStartBefore(newElement);
  range.setEndAfter(newElement);
  selection.removeAllRanges();
  selection.addRange(range);
}

// Add a listener for mouseup events on messages or hljs code blocks
$(document).on("mouseup", function (e) {
  // If "show-annotation-menu" is disabled, return
  if (!document.getElementById("show-annotation-menu").checked) return;
  // If the selection is without a monaco-editor, return
  if (!$(e.target).closest(".monaco-editor").length) return;
  let selection = window.getSelection();
  let selectedText = selection.toString();
  if (selectedText) {
    // Check if the selection is within a message or code block
    if (
      $(e.target).closest(
        ".message, pre, code, .language-python, .language-csharp"
      ).length
    ) {
      // Highlight selected text
      let range = selection.getRangeAt(0);

      // Replace range.surroundContents(span) with extractContents and insertNode
      let extractedContents = range.extractContents();
      span.appendChild(extractedContents);
      range.insertNode(span);

      // Make the annotation menu draggable
      $(".annotation-menu").draggable();

      saveSelection();
      $(".annotation-menu")
        .css({ top: e.pageY + 20, left: e.pageX })
        .show();
    }
  }
});

// Hide the annotation menu when clicking outside of it
$(document).click(function (e) {
  if (
    !$(e.target).closest(".annotation-menu").length &&
    !$(e.target).closest(".ui-autocomplete").length
  ) {
    if (!window.getSelection().toString()) {
      $(".annotation-menu").hide();
      savedSelection = null;
    }
  }
  // Collapse the column if clicking outside of it unless it's on the annotation menu or autocomplete list
  if (
    !$(e.target).closest(".solution-column").length &&
    !$(e.target).closest(".annotation-menu").length &&
    !$(e.target).closest(".ui-autocomplete").length
  ) {
    $(".solution-column").removeClass("expanded");
    $(".expand-button i").removeClass("fa-compress").addClass("fa-expand");
    adjustColumnWidths();
  }
});

// Initialize the annotation history array
let annotationHistory = [];

// Set up the autocomplete feature on the annotation input field
$("#annotation-input").autocomplete({
  source: function (request, response) {
    let term = request.term.toLowerCase();
    let matches = $.grep(annotationHistory, function (item) {
      return item.toLowerCase().indexOf(term) >= 0;
    });
    response(matches);
  },
  minLength: 0,
});

// Show the autocomplete suggestions when the input gains focus
$("#annotation-input").focus(function () {
  $(this).autocomplete("search", "");
});

$("#apply-annotation").click(function () {
  let annotationAction = $("#annotation-action").val();
  let annotationText = $("#annotation-input").val();

  if (!annotationText) {
    alert("Please enter annotation text.");
    return;
  }

  if (!activeEditor || !activeSelection) {
    alert("No text selected for annotation.");
    return;
  }

  const editor = activeEditor;
  const selection = activeSelection;
  const selectedText = editor.getModel().getValueInRange(selection);

  if (selectedText) {
    let annotatedText = "";

    let annotationType = $("#annotation-type").val();

    if (annotationType === "latex-inline") {
      annotatedText = `\\${annotationAction}[${annotationText}]{${selectedText}}`;
    } else if (annotationType === "HTML-inline") {
      annotatedText = `<${annotationAction} instruction="${annotationText}">${selectedText}</${annotationAction}>`;
    } else if (annotationType === "latex-id") {
      annotatedText = `[${annotationId}]{${selectedText}}`;
      let text = `[${annotationId}]: ${annotationAction} ${annotationText}`;
      annotationswithID += text + "\n";
      // Retrieve the id of monaco-editor
      let editorId = editor._domElement.id;
      editorId = editorId.replace("editor-", "");
      console.log("agent_columnId:", editorId);
      let comment_editor = window[`editorInstance_${editorId}`];
      console.log("editor:", comment_editor);
      // Add text in the comment editor
      comment_editor.setValue(annotationswithID);
      annotationId++;
    } else if (annotationType === "HTML-id") {
      annotatedText = `<${annotationId}>${selectedText}</${annotationId}>`;
      let text = `[${annotationId}]: ${annotationAction} ${annotationText}`;
      annotationswithID += text + "\n";
      // Retrieve the id of monaco-editor
      // Retrieve the id of monaco-editor
      let editorId = editor._domElement.id;
      editorId = editorId.replace("editor-", "");
      console.log("agent_columnId:", editorId);
      let comment_editor = window[`editorInstance_${editorId}`];
      console.log("editor:", comment_editor);
      // Add text in the comment editor
      comment_editor.setValue(annotationswithID);
      annotationId++;
    }

    // Replace the selected text with the annotated text
    editor.executeEdits("annotation", [
      {
        range: selection,
        text: annotatedText,
        forceMoveMarkers: true,
      },
    ]);

    // Apply decorations for visual feedback
    const decoration = {
      range: new monaco.Range(
        selection.startLineNumber,
        selection.startColumn,
        selection.endLineNumber,
        selection.endColumn
      ),
      options: {
        inlineClassName: "applied-annotation",
      },
    };
    editor.deltaDecorations([], [decoration]);

    // Hide the annotation menu
    $(".annotation-menu").hide();

    // Clear the global selection
    activeEditor = null;
    activeSelection = null;

    // Add to annotation history
    if (annotationText && !annotationHistory.includes(annotationText)) {
      annotationHistory.push(annotationText);
    }

    let actionDescription = `User applied annotation: action="${annotationAction}", text="${annotationText}"`;
    addHistoryEntry(actionDescription);
  }
});

$(".annotation-menu .close-btn").click(function () {
  $(".annotation-menu").hide();
  savedSelection = null;
});

$(document).on("click", ".menu-option", function () {
  // If the click is on a select element, send the selected value
  if ($(this).is("select")) {
    // If the id of the select element is llm-select, send a "H"
    if ($(this).attr("id") === "llm-select") {
      let option = $(this).find("option:selected").val();
      sendWebSocketMessage({ message: `H` });
      // Retrieve the text of the selected option
      selectedLLM = option;
    }

    // Hide the menu options after selecting one
    $("#menu-options").empty();
    // Hide all the menu options starting with menu-options-column-
    $("div[id^='menu-options-column-']").empty();
    // Retrieve the text of the selected option
    let text = $(this).find("option:selected").text();
    let actionDescription = `User selected menu option "${text}"`;
    addHistoryEntry(actionDescription);
    current_llm_in_use = $(this).find("option:selected").text();
    return;
  }
  let option = $(this).data("option");
  let input = $("#user-input").val();
  console.log("Sending option:", option);
  sendWebSocketMessage({ message: `${option}` });
  // Hide the menu options after selecting one
  $("#menu-options").empty();
  // Hide all the menu options starting with menu-options-column-
  $("div[id^='menu-options-column-']").empty();
  // Retrieve the text of the selected option
  let text = $(this).text();
  let actionDescription = `User selected menu option "${text}"`;
  addHistoryEntry(actionDescription);
});

$(document).on("click", ".regenerate", function () {
  let columnId = $(this).closest(".column").data("column-id");
  sendWebSocketMessage({ type: "regenerate", columnId: columnId });
});

$(document).on("click", ".delete", function () {
  $(this).closest(".column").remove();
});

$(
  ".filter-bar select, .filter-bar-bottom select, #search-filter, #search-filter-bottom"
).on("change keyup", function () {
  let messageType = $("#filter-message-type").val();
  let agentType = $("#filter-agent-type").val();
  let searchText = $("#search-filter").val().toLowerCase();

  let messageTypeBottom = $("#filter-message-type-bottom").val();
  let agentTypeBottom = $("#filter-agent-type-bottom").val();
  let searchTextBottom = $("#search-filter-bottom").val().toLowerCase();

  $("#message-accordion > div").each(function () {
    let $agentSection = $(this);
    let $agentHeader = $agentSection.children("h3");
    let $messages = $agentSection.find(".message");
    let $headers = $agentSection.find("h4");

    let agentVisible = false;
    let agentName = $agentHeader.text().toLowerCase();

    if (
      (agentType === "all" || agentName.includes(agentType.toLowerCase())) &&
      (agentTypeBottom === "all" ||
        agentName.includes(agentTypeBottom.toLowerCase()))
    ) {
      $messages.each(function (index) {
        let $message = $(this);
        let $header = $headers.eq(index);
        let headerText = $header.text().toLowerCase();
        let messageText = $message.text().toLowerCase();

        let showMessage =
          (messageType === "all" ||
            headerText.includes(messageType.toLowerCase())) &&
          (messageTypeBottom === "all" ||
            headerText.includes(messageTypeBottom.toLowerCase())) &&
          (messageText.includes(searchText) ||
            headerText.includes(searchText)) &&
          (messageText.includes(searchTextBottom) ||
            headerText.includes(searchTextBottom));

        $message.toggle(showMessage);
        $header.toggle(showMessage);

        if (showMessage) {
          agentVisible = true;
        }
      });
    }

    $agentSection.toggle(agentVisible);
  });

  $("#message-accordion").accordion("refresh");
});

// Toggle history sidebar
$("#toggle-history, #toggle-history-main").click(function () {
  $("#history-sidebar").toggle();
  $("#toggle-history-main").text(
    $("#history-sidebar").is(":visible") ? "Hide History" : "Show History"
  );
  $("#toggle-history").text(
    $("#history-sidebar").is(":visible") ? "Hide History" : "Show History"
  );
});

// Toggle settings sidebar
$("#toggle-settings, #toggle-settings-main").click(function () {
  $("#settings-sidebar").toggle();
  $("#toggle-settings-main").text(
    $("#settings-sidebar").is(":visible") ? "Hide Settings" : "Show Settings"
  );
  $("#toggle-settings").text(
    $("#settings-sidebar").is(":visible") ? "Hide Settings" : "Show Settings"
  );
});

// Synchronize message type filter between the two filter bars
$("#filter-message-type").change(function () {
  $("#filter-message-type-bottom").val($(this).val());
});

$("#filter-message-type-bottom").change(function () {
  $("#filter-message-type").val($(this).val());
});

// Synchronize agent type filter between the two filter bars
$("#filter-agent-type").change(function () {
  $("#filter-agent-type-bottom").val($(this).val());
});

$("#filter-agent-type-bottom").change(function () {
  $("#filter-agent-type").val($(this).val());
});

// When refresh the page, clear the columnsToDelete
localStorage.removeItem("columnsToDelete");
