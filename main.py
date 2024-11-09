import sys
import os
import pandas as pd
from openai import OpenAI

from agents.EvaluatorForBIRD import EvaluatorForBIRD
from agents.ZeroShotAgent import ZeroShotAgent
from agents.OptimizerAgent import OptimizerAgent
from agents.MultiAgentDiscussion import MultiAgentDiscussion

from config import MODEL, METHOD, OUTPUT_PATH
from utility import read_dataset, get_db_cursor, fetch_BIRD_schemas, get_openai_client, dump_to_json
import api_keys

os.environ['OPENAI_API_KEY'] = api_keys.OPENAI_API_KEY
sys.stdout = open(OUTPUT_PATH / 'output.txt', 'wt')   # logs all prints in the output directory



if METHOD == 'zero-shot':
    print(f"Experiment: {MODEL}_{METHOD}")
    
    # Setup
    df, db_names = read_dataset()
    db_schemas   = fetch_BIRD_schemas(db_names)
    print(f'{db_names=}, {len(df)=}')
    
    client = get_openai_client()
    agent = ZeroShotAgent(MODEL, client, get_db_cursor, db_schemas, OUTPUT_PATH)
    evaluator = EvaluatorForBIRD(get_db_cursor)
    
    # Generate
    raw_responses = agent.batched_generate(df)
    dump_to_json(f'{MODEL}_{METHOD}_raw_responses.json', raw_responses)

    # Parse
    print("Finished Generating. Attempting SQL auto-parsing...")
    cleaned_sql = agent.auto_parse_sql_from_response(raw_responses)
    dump_to_json(f'{MODEL}_{METHOD}_cleaned_sql.json', cleaned_sql)
    print("SQL auto-parsing successful")

    # Evaluate
    df['prediction'] = cleaned_sql
    df['label'] = evaluator.evaluate(df, pred_col_name='prediction')
    
    # Save results
    df.to_json(OUTPUT_PATH / f'{MODEL}_{METHOD}_df.json', orient='records')



if METHOD == 'optimizer-agent':
    print(f"Experiment: {MODEL}_{METHOD}")
    
    # Setup
    df, db_names = read_dataset()
    db_schemas   = fetch_BIRD_schemas(db_names)
    print(f'{db_names=}, {len(df)=}')
    
    client = get_openai_client()
    agent = OptimizerAgent(MODEL, client, get_db_cursor, db_schemas, OUTPUT_PATH)
    evaluator = EvaluatorForBIRD(get_db_cursor)
    
    # Generate
    df = pd.read_json('gpt-4o_zero-shot_df.json')
    raw_responses = agent.batched_generate(df)
    dump_to_json(f'{MODEL}_{METHOD}_raw_responses.json', raw_responses)

    # Parse
    print(f"Finished Generating. Attempting SQL auto-parsing...")
    cleaned_sql = agent.auto_parse_sql_from_response(raw_responses)
    dump_to_json(f'{MODEL}_{METHOD}_cleaned_sql.json', cleaned_sql)
    print(f"SQL auto-parsing successful")

    # Evaluate
    df['optimized'] = cleaned_sql
    df['opt-label'] = evaluator.evaluate(df, pred_col_name='optimized')
    
    # Save results
    df.to_json(OUTPUT_PATH / f'{MODEL}_{METHOD}_df.json', orient='records')



if METHOD == 'discussion':
    print(f"Experiment: {MODEL}_{METHOD}")
    
    # Setup
    df, db_names = read_dataset()
    db_schemas   = fetch_BIRD_schemas(db_names)
    print(f'{db_names=}, {len(df)=}')

    client = get_openai_client()
    multi_agent = MultiAgentDiscussion(MODEL, client, get_db_cursor, db_schemas, OUTPUT_PATH)
    evaluator = EvaluatorForBIRD(get_db_cursor)


    # Generate
    raw_responses = multi_agent.batched_generate(df, rounds=3)
    dump_to_json(f'{MODEL}_{METHOD}_raw_responses.json', raw_responses)

    # Parse
    print(f"Finished Generating. Attempting SQL auto-parse...")

    starter_zero = multi_agent.auto_parse_sql_from_response([response['agent_zero_shot'][0] for response in raw_responses])
    dump_to_json(f'{MODEL}_{METHOD}_cleaned_zeroshot_starter.json', starter_zero)

    starter_meta = multi_agent.auto_parse_sql_from_response([response['agent_meta_prompt'][0] for response in raw_responses])
    dump_to_json(f'{MODEL}_{METHOD}_cleaned_starter_meta.json', starter_meta)
    
    cleaned_sql  = multi_agent.auto_parse_sql_from_response([response['verdict'] for response in raw_responses])
    dump_to_json(f'{MODEL}_{METHOD}_cleaned_sql.json', cleaned_sql)

    print(f"SQL auto-parsing successful\n\n")


    # Evaluate results and save
    print("Evaluating Zero-shot starter generated queries")
    df['starter_zero_shot'] = starter_zero
    df['zero_shot_labels']  = evaluator.evaluate(df, pred_col_name='starter_zero_shot')

    print("Evaluating meta-prompt starter generated queries")
    df['starter_meta_prompt'] = starter_meta
    df['meta_prompt_labels']  = evaluator.evaluate(df, pred_col_name='starter_meta_prompt')

    print("Evaluating Multi-Agent Discussion generated queries")
    df['prediction'] = cleaned_sql
    df['label']      = evaluator.evaluate(df, pred_col_name='prediction')

    df.to_json(OUTPUT_PATH / f'{MODEL}_{METHOD}_df.json', orient='records')