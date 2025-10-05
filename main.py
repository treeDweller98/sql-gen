import datetime

import wandb

from sqlgen.experiments import (
    discuss_experiment,
    planner_exec_experiment,
    planner_plan_experiment,
    reasoner_picker_experiment,
    reasoner_zeroshot_experiment,
    zeroshot_experiment,
)
from utils import parse_args, read_bird_dataset, read_spider_dataset

if __name__ == "__main__":
    # Parse Arguments
    args = parse_args()
    # Create Output Directory
    args.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Read Dataset
    read_dataset = read_spider_dataset if args.DATASET == "spider" else read_bird_dataset
    df, databases = read_dataset(
        args.INPUT_DIR,
        args.QUESTION_FILENAME,
        args.DB_FOLDERNAME,
        args.USE_CACHED_SCHEMA,
        args.DB_EXEC_TIMEOUT,
        args.IS_DEBUG,
    )
    wandb.init(
        entity="mubtasimahasan",
        project="bappa-sql-logs",
        name=f"{args.EXPERIMENT}_{args.MODEL.value}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
    )

    start_time = datetime.datetime.now()
    print(f"Starting Experiment: {args.EXPERIMENT} with {args.MODEL.value} at {start_time}")
    match args.EXPERIMENT:
        case "zs":
            zeroshot_experiment(args, df, databases)
        case "rzs":
            reasoner_zeroshot_experiment(args, df, databases)
        case "mad":
            discuss_experiment(args, df, databases)
        case "pick":
            reasoner_picker_experiment(args, df, databases)
        case "plan":
            planner_plan_experiment(args, df, databases)
        case "plan-exec":
            planner_exec_experiment(args, df, databases)
        case _:
            print("INVALID EXPERIMENT SELECTED. ABORTING.")
    print(
        f"End of Experiment: {args.EXPERIMENT} with {args.MODEL.value}.\nTime taken: {datetime.datetime.now() - start_time}"
    )

    wandb.finish()
