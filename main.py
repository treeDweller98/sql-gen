import pandas as pd

from agents.ZeroShotAgent import ZeroShotAgent, MetaPromptZeroShotAgent
from agents.OptimizerAgent import OptimizerAgent
from agents.MultiAgentDiscussion import MultiAgentDiscussion
from agents.SchemaAugmenter import SchemaAugmenterAgent

from utils import initialise_experiment, cleanup_experiment, dump_to_json

if __name__ == '__main__':
    df, db_names, databases, llm, cfg, evaluator = initialise_experiment()

    agent_zs = ZeroShotAgent(llm, databases, dump_to_json)
    agent_mp = MetaPromptZeroShotAgent(llm, databases, dump_to_json)
    agent_op = OptimizerAgent(llm, databases, dump_to_json)

    try:
        # Zero-shot
        zs_raw = agent_zs.batched_generate(df, cfg)
        dump_to_json('zs_raw', zs_raw)
        zs_clean = agent_zs.batched_parse_sql(zs_raw)
        dump_to_json('zs_clean', zs_clean)

        df['pred_zs'] = zs_clean

        # Meta Prompt
        mp_raw = agent_mp.batched_generate(df, cfg)
        dump_to_json('mp_raw', mp_raw)
        mp_clean = agent_mp.batched_parse_sql(mp_raw)
        dump_to_json('mp_clean', mp_clean)

        df['pred_mp'] = mp_clean

        # Optimize Zero-shot
        op_zs_raw = agent_op.batched_generate(df, 'pred_zs', cfg)
        dump_to_json('op_zs_raw', op_zs_raw)
        op_zs_clean = agent_mp.batched_parse_sql(op_zs_raw)
        dump_to_json('op_zs_clean', op_zs_clean)

        df['pred_op_zs'] = op_zs_clean

        # Optimize Meta-prompt
        op_mp_raw = agent_op.batched_generate(df, 'pred_mp', cfg)
        dump_to_json('op_mp_raw', op_mp_raw)
        op_mp_clean = agent_op.batched_parse_sql(op_mp_raw)
        dump_to_json('op_mp_clean', op_mp_clean)

        df['pred_op_mp'] = op_mp_clean

        # Evaluate Performance
        zs_labels,    zs_report    = evaluator.evaluate(df, 'pred_zs')
        op_zs_labels, op_zs_report = evaluator.evaluate(df, 'pred_op_zs')
        mp_labels,    mp_report    = evaluator.evaluate(df, 'pred_mp')
        op_mp_labels, op_mp_report = evaluator.evaluate(df, 'pred_op_mp')

        df['zs_label']    = zs_labels
        df['op_zs_label'] = op_zs_labels
        df['mp_label']    = mp_labels
        df['op_mp_label'] = op_mp_labels

        results = '\n\n'.join([zs_report, op_zs_report, mp_report, op_mp_report])
        
    except Exception as e:
        results = f"Experiment Failed: {e.__class__.__name__} {e}. Check dumps."
    finally:
        cleanup_experiment(df, results)





# if METHOD == 'zero-shot':
#     print(f"Experiment: {MODEL}_{METHOD}")
    
#     # Setup
#     df, db_names = read_dataset()
#     db_schemas   = fetch_BIRD_schemas(db_names)
#     print(f'{db_names=}, {len(df)=}')
    
#     client = get_openai_client()
#     agent = ZeroShotAgent(MODEL, client, get_db_cursor, db_schemas, OUTPUT_PATH)
#     evaluator = EvaluatorForBIRD(get_db_cursor)
    
#     # Generate
#     raw_responses = agent.batched_generate(df)
#     dump_to_json(f'{MODEL}_{METHOD}_raw_responses.json', raw_responses)

#     # Parse
#     print("Finished Generating. Attempting SQL auto-parsing...")
#     cleaned_sql = agent.auto_parse_sql_from_response(raw_responses)
#     dump_to_json(f'{MODEL}_{METHOD}_cleaned_sql.json', cleaned_sql)
#     print("SQL auto-parsing successful")

#     # Evaluate
#     df['prediction'] = cleaned_sql
#     df['label'] = evaluator.evaluate(df, pred_col_name='prediction')
    
#     # Save results
#     df.to_json(OUTPUT_PATH / f'{MODEL}_{METHOD}_df.json', orient='records')



# if METHOD == 'optimizer-agent':
#     print(f"Experiment: {MODEL}_{METHOD}")
    
#     # Setup
#     df, db_names = read_dataset()
#     db_schemas   = fetch_BIRD_schemas(db_names)
#     print(f'{db_names=}, {len(df)=}')
    
#     client = get_openai_client()
#     agent = OptimizerAgent(MODEL, client, get_db_cursor, db_schemas, OUTPUT_PATH)
#     evaluator = EvaluatorForBIRD(get_db_cursor)
    
#     # Generate
#     df = pd.read_json('gpt-4o_zero-shot_df.json')
#     raw_responses = agent.batched_generate(df)
#     dump_to_json(f'{MODEL}_{METHOD}_raw_responses.json', raw_responses)

#     # Parse
#     print(f"Finished Generating. Attempting SQL auto-parsing...")
#     cleaned_sql = agent.auto_parse_sql_from_response(raw_responses)
#     dump_to_json(f'{MODEL}_{METHOD}_cleaned_sql.json', cleaned_sql)
#     print(f"SQL auto-parsing successful")

#     # Evaluate
#     df['optimized'] = cleaned_sql
#     df['opt-label'] = evaluator.evaluate(df, pred_col_name='optimized')
    
#     # Save results
#     df.to_json(OUTPUT_PATH / f'{MODEL}_{METHOD}_df.json', orient='records')



# if METHOD == 'discussion':
#     print(f"Experiment: {MODEL}_{METHOD}")
    
#     # Setup
#     df, db_names = read_dataset()
#     db_schemas   = fetch_BIRD_schemas(db_names)
#     print(f'{db_names=}, {len(df)=}')

#     client = get_openai_client()
#     multi_agent = MultiAgentDiscussion(MODEL, client, get_db_cursor, db_schemas, OUTPUT_PATH)
#     evaluator = EvaluatorForBIRD(get_db_cursor)


#     # Generate
#     raw_responses = multi_agent.batched_generate(df, rounds=3)
#     dump_to_json(f'{MODEL}_{METHOD}_raw_responses.json', raw_responses)

#     # Parse
#     print(f"Finished Generating. Attempting SQL auto-parse...")

#     starter_zero = multi_agent.auto_parse_sql_from_response([response['agent_zero_shot'][0] for response in raw_responses])
#     dump_to_json(f'{MODEL}_{METHOD}_cleaned_zeroshot_starter.json', starter_zero)

#     starter_meta = multi_agent.auto_parse_sql_from_response([response['agent_meta_prompt'][0] for response in raw_responses])
#     dump_to_json(f'{MODEL}_{METHOD}_cleaned_starter_meta.json', starter_meta)
    
#     cleaned_sql  = multi_agent.auto_parse_sql_from_response([response['verdict'] for response in raw_responses])
#     dump_to_json(f'{MODEL}_{METHOD}_cleaned_sql.json', cleaned_sql)

#     print(f"SQL auto-parsing successful\n\n")


#     # Evaluate results and save
#     print("Evaluating Zero-shot starter generated queries")
#     df['starter_zero_shot'] = starter_zero
#     df['zero_shot_labels']  = evaluator.evaluate(df, pred_col_name='starter_zero_shot')

#     print("Evaluating meta-prompt starter generated queries")
#     df['starter_meta_prompt'] = starter_meta
#     df['meta_prompt_labels']  = evaluator.evaluate(df, pred_col_name='starter_meta_prompt')

#     print("Evaluating Multi-Agent Discussion generated queries")
#     df['prediction'] = cleaned_sql
#     df['label']      = evaluator.evaluate(df, pred_col_name='prediction')

#     df.to_json(OUTPUT_PATH / f'{MODEL}_{METHOD}_df.json', orient='records')