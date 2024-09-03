from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
import re
import os

os.environ["OPENAI_API_KEY"] = "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"

# Set up the OpenAI LLM
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

# Helper function to check if the current chunk is satisfactory
def satisfied_with_chunk(output, state):
    # Add your logic here to determine if the chunk is satisfactory
    # For example, check if the output contains certain keywords or patterns
    if re.search(r'bad_word', output):
        return False
    return True

# Example usage
#prompt = "You always answer to a task by first indicating 3 alternative JSON format each corresponding to a different ontology and solving strategy, second by deciding which format you choose to use, then writing the story in the chosen JSON format. Task: Write a short story about a talking dog."
prompt = "You always answer in JSON following the format {'title':..., 'caracters':..., 'story':....}. Write a short story about a talking dog."
state = {}
output_chunks = []

for chunk in llm.stream(prompt,):
    #print(chunk.content, end='', flush=True)
    #print(f"New chunk:{chunk.content}")
    # if chunk.content contains ':' then ask user to input the value
    if ':' in chunk.content:
        print("\033[34m#\033[0mCurrent generation:" + ''.join(str(chunkstr) for chunkstr in output_chunks) + "\033[31m" + chunk.content +"\033[0m")
        user_input = input("Enter 'c' to cancel, 'r' to re-generate, or 'Enter' to continue: ")
    else:
        user_input = "#" # input("Enter 'c' to cancel, 'r' to re-generate, or 'Enter' to continue: ")

    if user_input.lower() == 'c':
        # Cancel the current chunk
        state = {}
        output_chunks = []
        print("Canceled.")
    elif user_input.lower() == 'r':
        # Re-generate the current chunk
        state = {}
        output_chunks = []
        print("Re-generating...")
    else:
        output_chunks.append(chunk.content)
        # print the current output
        #print("Total generated:" + '_'.join(str(chunk) for chunk in output_chunks))

output = ''.join(str(chunk) for chunk in output_chunks)
print("\nFinal output:")
print(output)