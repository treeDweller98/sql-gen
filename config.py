### config.py
from enum import Enum
from pathlib import Path

class SupoortedModels(Enum):
    qwen25_coder_3b_instruct_awq  = 'Qwen/Qwen2.5-Coder-3B-Instruct-AWQ'
    qwen25_coder_14b_instruct_awq = 'Qwen/Qwen2.5-Coder-14B-Instruct-AWQ'
    qwen25_coder_14b_instruct_gptq_int4 = 'Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4'
    # Add more here


### Experiment Configurations ###
### Import only to utils.py   ###
EXPERIMENT = [
    'mad',
    'zero-meta-optim-unaug',
][0]

MODEL = SupoortedModels.qwen25_coder_14b_instruct_awq.value       # TODO: make this settable from bash somehow maybe?
GPU_MEMORY_UTILIZATION = 0.97
TENSOR_PARALLEL_SIZE = 2                # set equal to number of GPU               
MODEL_MAX_SEQ_LEN = 4096 * 1            # 4096*2 is max for 14B AWQ model with fp8 KV-cache on 15GB VRAM 
KV_CACHE_DTYPE = 'fp8'                  # Reduces memory consumption; fp8 might impact models that use Grouped Query Attn like Qwen
BATCH_SIZE = 8                          # saves after every batch
SEED = 42


INPUT_PATH  = Path(f'/kaggle/working/bird-bench/bird-bench/bird-minidev')
OUTPUT_PATH = Path(f'/kaggle/working/results/{MODEL}_{EXPERIMENT}/')
BIRD_QUESTION_FILENAME = 'dev.json'
DATABASES_FOLDERNAME = 'dev_databases'
DB_EXEC_TIMEOUT = 30.0                              # maximum number of seconds a query execution is allowed to take
USE_CACHED_SCHEMA = None #Path('/kaggle/working/bird-bench/aug-minidev/aug-minidev/aug.json')  # Use pre-generated schema 

# set all to FALSE for actual runs
USE_DEBUG_DATASET = True                           # Debug with only first 15 bird questions
USE_DEBUG_DB = False                               # True for ['formula_1', 'debit_card_specializing', 'thrombosis_prediction'] only subset