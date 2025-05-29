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
// Function to get URL parameters safely
function getUrlParameter(name) {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get(name) || null;
}
function generateRequestId() {
  return "_" + Math.random().toString(36).substr(2, 9);
}
// Function to validate WebSocket URLs
function isValidWebSocketUrl(url) {
  try {
    const parsedUrl = new URL(url);
    return parsedUrl.protocol === "ws:" || parsedUrl.protocol === "wss:";
  } catch (e) {
    return false;
  }
}

function removeUnicodeSequences(jsonString) {
  // Expression régulière pour trouver les séquences \uXXXX
  const unicodeRegex = /\\u[\dA-Fa-f]{4}/g;

  // Remplacer toutes les occurrences par une chaîne vide
  return jsonString.replace(unicodeRegex, "");
}
// Function to generate an UUID
function generateUUID() {
  // Générateur d'UUID version 4
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0,
      v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Function to get an unique identifier for this device
function getUniqueIdentifier() {
  let uniqueId = localStorage.getItem("uniqueId");
  if (!uniqueId) {
    uniqueId = generateUUID();
    localStorage.setItem("uniqueId", uniqueId);
  }
  return uniqueId;
}

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

