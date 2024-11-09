# Agent-based Text-to-SQL on BIRD-bench

## Overview
This repository contains Python code for running Text-to-SQL generation experiments on the BIRD-bench dataset.
Text generation uses the OpenAI Chat Completion API. Works with OpenAI as well as locally hosted Ollama models.

## Instructions
1. Clone repository and run the installer:
    ```bash
    git clone https://github.com/<REPOSITORY>`
    cd path/to/sql-gen
    sh install.sh
    ``` 
1. Download and unpiz datasets.
    - train: https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
    - dev: https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
    - minidev: https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
1. Place BIRD dataset in `./data/`, ensuring that the following structure is followed:
    ```
    +-- data
    |   +-- bird-minidev
    |       +-- dev.json
    |       +-- dev_databases
    |           +-- california_schools
    |               +-- database_descriptions
    |               +-- california_schools.sqlite
    |           +--              ...
    ```
    The functions in `utility.py` assume the existence of `dev.json` and `dev_databases` in the directory `INPUT_PATH`; rename files and directories as required. 
1. Place your OPENAI_API_KEY and other secrets in `api_keys.py`.
1. Configure experiment with `config.py`.
1. Run experiment with `main.py` or `sql-gen.ipynb`