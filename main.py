from vllm import LLM, SamplingParams
from utils import read_dataset, parse_args
from experiments import (
    zeroshot_experiment,
    reasoner_zeroshot_experiment,
    discuss_experiment,
    debate_experiment,
)


def setup_experiment():
    # Parse Arguments
    args = parse_args()
    # Read Dataset
    df, databases = read_dataset(
        args.INPUT_PATH, args.BIRD_QUESTION_FILENAME, args.DB_FOLDERNAME, 
        args.USE_CACHED_SCHEMA, args.DB_EXEC_TIMEOUT
    )
    # LLM and Default Generation Config
    llm = LLM(
        args.MODEL.value,
        gpu_memory_utilization=args.GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=args.TENSOR_PARALLEL_SIZE,
        max_model_len=args.MODEL_MAX_SEQ_LEN,
        kv_cache_dtype=args.KV_CACHE_DTYPE,
        seed=args.SEED,
        dtype=args.VLLM_DTYPE,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    # Create Output Directory
    args.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    return df, databases, llm, args



if __name__ == '__main__':
    df, databases, llm, args = setup_experiment()
    experiment = args.EXPERIMENT

    print(f"Starting Experiment: {experiment} with {args.MODEL.value}")

    match experiment:
        case 'zs':
            cfg = SamplingParams(
                temperature=0,
                top_p=1,
                repetition_penalty=1.05,
                max_tokens=1024,
            )
            zeroshot_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)
        case 'rzs':
            cfg = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=30,
                repetition_penalty=1.0,
                max_tokens=4096*2,
            )
            reasoner_zeroshot_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)
        case 'mad':
            cfg = SamplingParams(
                temperature=0,
                top_p=1,
                repetition_penalty=1.05,
                max_tokens=1024*2,
            )
            discuss_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)
        case 'madb':
            cfg = SamplingParams(
                temperature=0,
                top_p=1,
                repetition_penalty=1.05,
                max_tokens=1024*2,
            )
            debate_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)
        case _:
            print("INVALID EXPERIMENT SELECTED. ABORTING.")
    
    print(f"End of Experiment: {experiment} with {args.MODEL.value}")
