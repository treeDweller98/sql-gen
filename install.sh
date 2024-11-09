#!/bin/sh

# Setup environments with required packages
python3 -m venv .venv
. "$PWD/.venv/bin/activate"
python3 -m pip install pandas tqdm openai ollama

# Create secrets file
printf "OPENAI_API_KEY = ''    # your api key here" > api_key.py

# Make dataset directory
mkdir data
# TODO: add automatic download of dataset and renaming