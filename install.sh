#!/bin/sh

# Setup environment with required packages
python3 -m venv .venv
. "$PWD/.venv/bin/activate"
python3 -m pip install pandas tqdm openai ollama

curl -fsSL https://ollama.com/install.sh | sh

# Create secrets file
printf "OPENAI_API_KEY = ''    # your api key here" > api_key.py

# Make dataset directory
mkdir data
mkdir results
# TODO: add automatic download of dataset and renaming