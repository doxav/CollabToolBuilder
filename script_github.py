import os
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()
source_dir =os.getenv("SOURCE_DIR")
dest_dir = os.getenv("DEST_DIR")
post_git_script = os.getenv("POST_GIT_SCRIPT")
post_git = os.getenv("POST_GIT")

def execute_git_commands():
    try:
        for file_name in os.listdir(source_dir):
            file_path = os.path.join(source_dir, file_name)
            if os.path.isfile(file_path):
                destination_path = os.path.join(dest_dir, file_name)
                shutil.copy2(file_path, destination_path)
                print(f"File {file_name} copied to {dest_dir}")

                subprocess.run(["git", "-C", dest_dir, "add", file_name], check=True)
                subprocess.run(["git", "-C", dest_dir, "commit", "-m", f"Add or update {file_name}"], check=True)
                subprocess.run(["git", "-C", dest_dir, "push"], check=True)

                print(f"File {file_name} added, committed, and pushed to the remote repository.")

    except Exception as e:
        print(f"Error during Git operations: {e}")


def run_start_script():
    
    working_directory = post_git
    print("working dir", working_directory)
    try:
        result = subprocess.run(["bash", post_git_script], cwd=working_directory, capture_output=True, text=True)
        
        # Output the result
        print("start.sh has been reload")

        if result.returncode != 0:
            print("Script Error:")
            print(result.stderr)
        else:
            print("Script executed successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    execute_git_commands()
    run_start_script()