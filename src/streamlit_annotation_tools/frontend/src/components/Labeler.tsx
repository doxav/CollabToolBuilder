import React, { useReducer, useEffect, useCallback, useState } from "react";
import { Streamlit } from "streamlit-component-lib";
import { useRenderData } from "../utils/StreamlitProvider";
import { ActionTypes, IAction, IState } from "../types/labelerTypes";
import { initialState, reducer } from "../reducers/labelerReducer";
import { adjustSelectionBounds, getCharactersCountUntilNode, isLabeled, removeLabelData } from "../helpers/labelerHelpers";
import './styles.css'; // Import the CSS file

const highlightPythonSyntax = (text: string): string => {
  const keywords = /\b(import|from|as|def|class|return|if|else|elif|for|while|try|except|finally|with|yield|lambda|global|nonlocal|assert|break|continue|pass|raise|del|not|or|and|is|in)\b/g;
  const strings = /(\".*?\"|\'.*?\')/g;
  const comments = /(#.*?$)/gm;

  // Escape < and > characters
  const escapedText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');

  return escapedText
    .replace(keywords, '<span class=python-keyword>$&</span>')
    .replace(strings, '<span class=python-string>$&</span>')
    .replace(comments, '<span class=python-comment>$&</span>');
};

const Labeler: React.FC = () => {
  const { args } = useRenderData();
  const [labelName, setLabelName] = useState<string>("");
  const [prefix, setPrefix] = useState<string>("keep");
  const [selectedLabelId, setSelectedLabelId] = useState<string>("");
  const [state, dispatch] = useReducer<React.Reducer<IState, IAction>>(reducer, initialState);

  useEffect(() => {
    const fetchData = async () => {
      const { text, labels, in_snake_case } = args;
      const highlightedText = highlightPythonSyntax(text);
      dispatch({ type: ActionTypes.SET_TEXT_LABELS, payload: { text: highlightedText, labels, in_snake_case } });
      dispatch({ type: ActionTypes.RENDER_TEXT });
      Streamlit.setComponentValue(labels);
    };

    fetchData();
  }, [args]);

  useEffect(() => {
    dispatch({ type: ActionTypes.RENDER_TEXT });
  }, [state.labels, state.selectedLabel]);

  const handleMouseUp = useCallback(async () => {
    if (!state.selectedLabel) return;
    const selection = document.getSelection()?.getRangeAt(0);

    if (selection && selection.toString().trim() !== "") {
      const container = document.getElementById("actual-text");
      const charsBeforeStart = getCharactersCountUntilNode(selection.startContainer, container);
      const charsBeforeEnd = getCharactersCountUntilNode(selection.endContainer, container);

      const finalStartIndex = selection.startOffset + charsBeforeStart;
      const finalEndIndex = selection.endOffset + charsBeforeEnd;

      const textContent = container?.textContent || "";

      const { start, end } = adjustSelectionBounds(textContent, finalStartIndex, finalEndIndex);
      const selectedText = textContent.slice(start, end);

      if (isLabeled(finalStartIndex, finalEndIndex, state.labels[state.selectedLabel])) {
        const labels = removeLabelData(start, end, state.labels[state.selectedLabel]);
        const newLabels = { ...state.labels };
        newLabels[state.selectedLabel] = labels;
        dispatch({ type: ActionTypes.SET_TEXT_LABELS, payload: { text: state.text, labels: newLabels, in_snake_case: state.in_snake_case } });
      } else {
        const label = { start, end, label: selectedText };
        const newLabels = { ...state.labels };
        newLabels[state.selectedLabel] = [...newLabels[state.selectedLabel], label];
        dispatch({ type: ActionTypes.SET_TEXT_LABELS, payload: { text: state.text, labels: newLabels, in_snake_case: state.in_snake_case } });
      }
    }
  }, [state, dispatch]);

  const addLabel = (name: string) => {
    if (name.trim() === "") return;

    let finalLabelName = name;
    if (prefix === "replace by" && selectedLabelId) {
      finalLabelName = `${prefix}: ${name} (${selectedLabelId})`;
    } else {
      finalLabelName = `${prefix}: ${name}`;
    }

    setLabelName("");
    setSelectedLabelId("");
    dispatch({ type: ActionTypes.ADD_LABEL, payload: finalLabelName });
  };

  const selectLabel = (name: string) => {
    dispatch({ type: ActionTypes.SELECT_LABEL, payload: name });
  };

  const removeLabel = (name: string) => {
    dispatch({ type: ActionTypes.REMOVE_LABEL, payload: name });
  };

  return (
    <div>
      <div className="flex flex-row flex-wrap">
        <div className="flex flex-wrap justify-between items-center cursor-pointer mr-2 mb-2 pr-3 rounded text-white text-base bg-primary hover:bg-secondary">
          <input
            type="text"
            placeholder="Enter Label Name"
            className="text-black p-1 mr-2 focus:outline-none border border-secondary"
            onChange={(e) => setLabelName(e.target.value)}
            value={labelName}
          />
          <select
            className="text-black p-1 mr-2 focus:outline-none border border-secondary"
            value={prefix}
            onChange={(e) => setPrefix(e.target.value)}
          >
            <option value="keep">keep</option>
            <option value="re-generate">re-generate</option>
            <option value="replace by">replace by</option>
          </select>
          {prefix === "replace by" && (
            <select
              className="text-black p-1 mr-2 focus:outline-none border border-secondary"
              value={selectedLabelId}
              onChange={(e) => setSelectedLabelId(e.target.value)}
            >
              <option value="">Select Label ID</option>
              {Object.keys(state.labels).map((label, index) => (
                <option key={index} value={label}>
                  {label}
                </option>
              ))}
            </select>
          )}
          <button onClick={() => addLabel(labelName)}>Add Label</button>
        </div>

        {Object.keys(state.labels).map((label, index) => (
          <span
            key={index}
            className={
              "flex flex-wrap justify-between items-center cursor-pointer py-1 px-3 mr-2 mb-2 rounded text-base" +
              (state.selectedLabel === label
                ? " bg-primary hover:bg-secondary text-white"
                : " border border-primary text-primary hover:bg-primary hover:text-white")
            }
            onClick={() => selectLabel(label)}
          >
            {label}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 ml-3 hover:text-gray-300"
              viewBox="0 0 20 20"
              fill="currentColor"
              onClick={() => removeLabel(label)}
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 000 16zM8.707 7.293a1 1 000-1.414 1.414L8.586 10l-1.293 1.293a1 1 101.414 1.414L10 11.414l1.293 1.293a1 1 001.414-1.414L11.414 10l1.293-1.293a1 1 00-1.414-1.414L10 8.586 8.707 7.293z"
              />
            </svg>
          </span>
        ))}
      </div>
      <div id="actual-text" className="mt-5 h-full python-code" onMouseUp={handleMouseUp} dangerouslySetInnerHTML={{ __html: state.text }}>
      </div>
    </div>
  );
};

export default Labeler;
