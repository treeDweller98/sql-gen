from pathlib import Path
from core.Model import SupportedModels


### Experiment Configurations ###
EXPERIMENT = [
    'zeroshot-metaprompt-optimizer'
][0]
MODEL = SupportedModels.Ollama.llama_32_1b
IS_INFERENCING_LOCALLY = True                               # Deploys ollama locally if IS_LOCAL
MODEL_DEBUG_MODE = False                                    # Models don't generate text in debug mode

OUTPUT_PATH = Path(f'results/{MODEL.value}_{EXPERIMENT}/')
INPUT_PATH  = Path('data/bird-minidev/')

BIRD_QUESTION_FILENAME = 'dev.json'
DATABASES_FOLDERNAME = 'dev_databases'
USE_CACHED_SCHEMA = False                                   # Use pre-generated schema instead of augmenting with LLM from scratch
USE_FULL_DB = False                                         # False for small development subset
USE_DEBUG_DATASET = False                                   # debug with only first 5 bird questions

IS_PRINT_TO_FILE = False                                    # If True: sends print() to OUTPUT_PATH/output.txt instead of sys.stdout

SEED = 42