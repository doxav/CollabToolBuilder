import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Répertoire à surveiller
directory_to_watch = "/home/hadrien/Documents/CollabFunctionsGPTCreator/pipeline_create"
branch_name = "Jira-FA"  # Changez selon votre branche
commit_message = "Commit of a new file detected by the loop"

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        
        if event.is_directory:
            return None
        elif event.src_path.endswith(".py"):
            print(f"New file detected: {event.src_path}")
            self.git_add_commit_push(event.src_path)

    def git_add_commit_push(self, file_path):
        
        subprocess.run(["git", "add", file_path])
        
        subprocess.run(["git", "commit", "-m", commit_message])
        
        subprocess.run(["git", "push", "origin", branch_name])
        print(f"file {file_path} has been push on GitHub.")

if __name__ == "__main__":
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
