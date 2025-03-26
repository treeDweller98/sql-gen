import datetime
from vllm import LLM, SamplingParams
from utils import read_dataset, parse_args, load_llm, del_llm
from experiments import (
    zeroshot_experiment,
    reasoner_zeroshot_experiment,
    discuss_experiment,
    debate_experiment,
    reasoner_picker_experiment,
)


if __name__ == '__main__':
    # Parse Arguments
    args = parse_args()
    # Create Output Directory
    args.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    # Read Dataset
    df, databases = read_dataset(
        args.INPUT_PATH, args.BIRD_QUESTION_FILENAME, args.DB_FOLDERNAME, 
        args.USE_CACHED_SCHEMA, args.DB_EXEC_TIMEOUT, args.IS_DEBUG,
    )

    start_time = datetime.datetime.now()
    print(f"Starting Experiment: {args.EXPERIMENT} with {args.MODEL.value} at {start_time}")

    match args.EXPERIMENT:
        case 'zs':
            llm = load_llm(args)
            cfg = SamplingParams(
                temperature=0,
                top_p=1,
                repetition_penalty=1.05,
                max_tokens=1024,
            )
            zeroshot_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)

        case 'rzs':
            llm = load_llm(args)
            cfg = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=30,
                repetition_penalty=1.0,
                max_tokens=4096*2,
            )
            reasoner_zeroshot_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)

        case 'mad':
            llm = load_llm(args)
            cfg = SamplingParams(
                temperature=0,
                top_p=1,
                repetition_penalty=1.05,
                max_tokens=1024*2,
            )
            discuss_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)

        case 'madb':
            llm = load_llm(args)
            cfg = SamplingParams(
                temperature=0,
                top_p=1,
                repetition_penalty=1.05,
                max_tokens=1024*2,
            )
            debate_experiment(df, databases, llm, cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)

        # case 'pick':
        #     rzn_cfg = SamplingParams(
        #         temperature=0.6,
        #         top_p=0.95,
        #         top_k=30,
        #         repetition_penalty=1.0,
        #         max_tokens=4096*2,
        #     )
        #     code_cfg = SamplingParams(
        #         temperature=0,
        #         top_p=1,
        #         repetition_penalty=1.05,
        #         max_tokens=1024,
        #     )
        #     reasoner_picker_experiment(df, databases, rzn_llm, code_llm, rzn_cfg, code_cfg, args.OUTPUT_PATH, args.BATCH_SIZE, args.EXPERIMENT)
        
        case _:
            print("INVALID EXPERIMENT SELECTED. ABORTING.")

    print(f"End of Experiment: {args.EXPERIMENT} with {args.MODEL.value}.\nTime taken: {datetime.datetime.now() - start_time}")
