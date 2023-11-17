#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

# Download the output.mov file from Google Drive
gdown -O "$DIR/data/traffic_analysis.mov" "https://drive.google.com/uc?id=1oW1Ryo0qAAS2NHmWCjVQfpcdlHt-gXnj"

# Download the model4.pt file from Google Drive
gdown -O "$DIR/data/traffic_analysis.pt" "https://drive.google.com/uc?id=1EZ7G1nTG6ezWZZGteuvmnOlCuLz7i0R2"
