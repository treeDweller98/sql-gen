from utils import setup_experiment, mad_experiment

if __name__ == '__main__':
    df, databases, cfg, llm = setup_experiment()
    mad_experiment(df, databases, llm, 'multiag')
    