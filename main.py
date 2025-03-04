from vllm import LLM, SamplingParams
from utils import read_dataset, parse_args
from experiments import zeroshot_experiment


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
        max_seq_len_to_capture=args.MODEL_MAX_SEQ_LEN,
        kv_cache_dtype=args.KV_CACHE_DTYPE,
        seed=args.SEED,
        dtype=args.VLLM_DTYPE,
        trust_remote_code=True,
    )
    cfg = SamplingParams(
        temperature=0,
        top_p=1,
        repetition_penalty=1.1,
        max_tokens=4096,
    )
    # Create Output Directory
    args.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    return df, databases, cfg, llm, args



if __name__ == '__main__':
    df, databases, cfg, llm, args = setup_experiment()
    print(f"Starting Experiment {args.EXPERIMENT} with {args.MODEL.value}")
    zeroshot_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)