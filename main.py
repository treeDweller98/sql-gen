import datetime
from utils import read_dataset, parse_args
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
            zeroshot_experiment(args, df, databases)
        case 'rzs':
            reasoner_zeroshot_experiment(args, df, databases)
        case 'mad':
            discuss_experiment(args, df, databases)
        case 'madb':
            debate_experiment(args, df, databases)
        case 'pick':
            reasoner_picker_experiment(args, df, databases)
        case _:
            print("INVALID EXPERIMENT SELECTED. ABORTING.")

    print(f"End of Experiment: {args.EXPERIMENT} with {args.MODEL.value}.\nTime taken: {datetime.datetime.now() - start_time}")
