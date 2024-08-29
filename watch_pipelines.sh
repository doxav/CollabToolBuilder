

PIPELINES_DIR="/home/hadrien/Documents/CollabFunctionsGPTCreator/pipelines/pipelines"
START_SCRIPT="/home/hadrien/Documents/CollabFunctionsGPTCreator/pipelines/start.sh"

inotifywait -m -e create,delete,modify $PIPELINES_DIR | while read path action file; do
    echo "Detected $action on $file. Running start.sh..."

    python script_github.py
    echo "script watchhhh activé"
    $START_SCRIPT
done

