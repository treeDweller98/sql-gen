from pathlib import Path

### Experiment Configurations ###
SEED = 42
METHOD = [
    'zero-shot',
    'optimizer-agent',
    'discussion',
][2]
MODEL = [
    'gpt-4o',
    'gpt-4o-mini',
    'llama3.1',
][0]
INPUT_PATH  = Path('data/bird-minidev/')
OUTPUT_PATH = Path(f'results/{MODEL}_{METHOD}/')
IS_USE_FULL_DB = False                              # False for small development subset