import re

text1n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

Success: False
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text1y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

Success: True
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text2n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

success=no
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text2y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

success=yes
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text3y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

success='yes'
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text3n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

success='no'
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text4n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

  SUCCESS = FALSE
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text4y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

  SUCCESS = TRUE
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text5n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

success:='false'
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text5y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

success:='true'
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text6n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success":false
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text6y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success":true
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text7n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success": "False"
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text7y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success": "True"
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text8n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success": 0
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text8y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success": 1
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text9n = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success": n
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text9y = """The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

"success": y
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success."""

text10n = """
The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

[Success: 0]
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success.
"""

text10y = """
The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

[Success: 1]
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success.
"""

text11n = """
The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

(Success: No)
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success.
"""

text11y = """
The code seems to be structured to achieve the task's goals by utilizing regular expressions for section identification and a text summarization function for summarizing the content. However, the runtime errors at execution prevent a direct assessment of the code's functionality.

(Success: Yes)
Explain: The code's runtime errors prevent a full evaluation of its success. Without the ability to execute the code, it's unclear if it accurately identifies and summarizes the key sections from scientific papers. Additionally, it's essential to ensure that the NLP tools mentioned in the plan are effectively integrated into the code for accurate section identification and summarization. Further debugging and testing are required to determine the code's success.
"""

def get_success_value_in_text(text):
    match = re.search(r"Success['\"]?\s*[:=][:=]?\s*(['\"]?)(True|False|Yes|No|y|n|0|1)\1", text, re.IGNORECASE)
    if match:
        success_value = match.group(2)
        return success_value.lower() in ['true', 'yes', 'y', '1']
    return False

for id, text in enumerate([text1n, text1y, text2n, text2y, text3n, text3y, text4n, text4y, text5n, text5y, text6n, text6y, text7n, text7y, text8n, text8y, text9n, text9y, text10n, text10y, text11n, text11y]):
    is_success = get_success_value_in_text(text)
    print(f"Success{id/2}: {is_success}")