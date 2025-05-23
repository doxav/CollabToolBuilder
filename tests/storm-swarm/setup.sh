#!/bin/bash

# Step 1: Run the SearXNG server as a Docker container
docker run --rm -d -p 8080:8080 -v "$(pwd)/searxng:/etc/searxng" \
    -e "BASE_URL=http://localhost:8080/" \
    -e "INSTANCE_NAME=my-instance" \
    searxng/searxng

# Step 2: Install dependency packages
echo "Installing dependencies..."b
pip install -r requirements.txt

# Step 3: Clone the GitHub Repository
echo "Cloning the GitHub repository..."
git clone https://github.com/stanford-oval/storm.git


# Step 4: Check if secrets.toml contains a valid OPENAI_API_KEY
SECRETS_FILE="secrets.toml"
if [ -f "$SECRETS_FILE" ]; then
    # Extract the value of OPENAI_API_KEY
    OPENAI_API_KEY=$(grep -oP '^OPENAI_API_KEY="\K[^"]*' "$SECRETS_FILE")
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY is empty or not set in $SECRETS_FILE."
        exit 1
    elif [[ "$OPENAI_API_KEY" == *"{{"* || "$OPENAI_API_KEY" == *"}}"* ]]; then
        echo "Error: OPENAI_API_KEY contains a template string in $SECRETS_FILE."
        exit 1
    else
        echo "Valid OPENAI_API_KEY found in $SECRETS_FILE."
        echo $OPENAI_API_KEY
        export OPENAI_API_KEY=$OPENAI_API_KEY
        echo "OPENAI_API_KEY exported as an environment variable."
    fi
else
    echo "Error: $SECRETS_FILE does not exist."
    exit 1
fi

# Step 5: Copy secrets.toml to the storm directory
cp secrets.toml storm/

# Step 6: Apply patch
cd storm
git apply ../integrate_storm.patch