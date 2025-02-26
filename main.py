from utils import setup_experiment, mad_experiment, zeroshot_experiment


if __name__ == '__main__':
    df, databases, cfg, llm = setup_experiment()
    zeroshot_experiment(df, databases, llm, cfg, 'zs')
    # mad_experiment(df, databases, llm, 'multiag')