from typing import Literal, Callable
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from sqlgen.base_agent import TextToSQL, TextToSQLGenerationOutput
from core.dbhandler import SQLiteDatabase


class ZeroShotStarter(TextToSQL):
    personas = {
        'simple': "who offers short, and simple solutions to user questions",
        'technical': "who provides highly technical answers to user questions",
        'thinker': "who does not hesistate to dig deep into a problem and explore several approaches before settling on a solution"
    }

    def process_bird_df(self, idx: int, row: pd.DataFrame, persona: Literal['simple', 'technical', 'thinker']) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, self.personas[persona]

    def generate_prompt(self, schema: str, question: str, persona: str) -> str:
        prompt = (
            f"You are a helpful SQL coding assistant{' ' + persona if persona else ''}. "
            "Please generate a SQLite query to answer the user question, based on the schema below. "
            "In your response, first briefly explain your reasoning. Your final answer should be enclosed "
            "in a markdown code block.\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### QUESTION\n{question}.\n\n"
            f"### RESPONSE\nLet's think step by step "
        )
        return prompt


class DiscussionAgent(TextToSQL):
    # TODO: Add personas to DiscussionAgent
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple[str, str, dict[int, str]]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        other_responses = ''.join(
            f"###### Agent_{agent}\n{response}\n\n"
            for agent, response in agent_responses.items()
        )
        prompt = (
            "You are a helpful SQL coding agent. You understand that collaborative discussion is the best way to solve problems. "
            "Using the other agent's response as additional information, your job is to generate a SQLite query to answer the "
            "user question based on the schema. In your response, please explain your reasoning clearly so that others may "
            "give you constructive feedback. Your final answer should be enclosed in a markdown code block.\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### AGENT RESPONSES\n{other_responses}"
            f"### QUESTION\n{question}\n\n"
            f"### YOUR RESPONSE\nLet's think step by step "
        )
        return prompt
    

class DiscussionJudge(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        n_agents = len(agent_responses)
        other_responses = ''.join(
            f"###### Agent_{agent}\n{response}\n\n"
            for agent, response in agent_responses.items()
        )
        prompt = (
            f"You are a SQL expert overseeing {n_agents} coding agents collaborating to answer the user question based on the given schema. "
            f"Using the other agents' responses as additional information, generate the production-ready SQLite query. "
            f"Your final answer should be enclosed in a markdown code block.\n\n"
            f"### QUESTION\n{question}\n\n"
            f"### SCHEMA\n{schema}\n\n"
            f"### AGENT RESPONSES\n{other_responses}"
            f"### QUESTION\n{question}\n\n"
            f"### VERDICT\nLet's think step by step "
        )
        return prompt


class MultiAgentDiscussion:
    def discuss(
        df: pd.DataFrame, databases: dict[str, SQLiteDatabase], llm: LLM,
        output_path: Path, savename: str, batch_size: int, evaluator_fn: Callable
    ) -> pd.DataFrame:
            
        # TODO: Add personas to DiscussionAgent
        starter = ZeroShotStarter(llm, databases, output_path)
        agent_1 = DiscussionAgent(llm, databases, output_path)
        agent_2 = DiscussionAgent(llm, databases, output_path)
        agent_3 = DiscussionAgent(llm, databases, output_path)
        judge   = DiscussionJudge(llm, databases, output_path)
        
        cfg = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.2,
            max_tokens=2048,
        )
        
        def gather_agent_responses(
                agent_num1: int, responses_1: TextToSQLGenerationOutput, 
                agent_num2: int, responses_2: TextToSQLGenerationOutput,
            ) -> list[dict[int, str]]:
            agent_responses = [
                {agent_num1: resp1, agent_num2: resp2}
                for resp1, resp2 in zip(responses_1.raw_responses, responses_2.raw_responses)
            ]
            return agent_responses
        
        def gather_all_responses(
                responses_1: TextToSQLGenerationOutput, 
                responses_2: TextToSQLGenerationOutput, 
                responses_3: TextToSQLGenerationOutput,
            ) -> list[dict[int, str]]:
            agent_responses = [
                {1: resp1, 2: resp2, 3: resp3}
                for resp1, resp2, resp3 in zip(responses_1.raw_responses, responses_2.raw_responses, responses_3.raw_responses)
            ]
            return agent_responses
        
        starters_1, starters_1_label = starter.batched_generate(df, cfg, batch_size, 'starter1', evaluator_fn, persona='simple')
        starters_2, starters_2_label = starter.batched_generate(df, cfg, batch_size, 'starter2', evaluator_fn, persona='technical')
        starters_3, starters_3_label = starter.batched_generate(df, cfg, batch_size, 'starter3', evaluator_fn, persona='thinker')

        agent_1_discuss_r1, a1r1_label = agent_1.batched_generate(df, cfg, batch_size, f'agent1_R{1}', evaluator_fn, agent_responses=gather_agent_responses(2, starters_2, 3, starters_3))
        agent_2_discuss_r1, a2r1_label = agent_2.batched_generate(df, cfg, batch_size, f'agent2_R{1}', evaluator_fn, agent_responses=gather_agent_responses(1, starters_1, 3, starters_3))
        agent_3_discuss_r1, a3r1_label = agent_3.batched_generate(df, cfg, batch_size, f'agent3_R{1}', evaluator_fn, agent_responses=gather_agent_responses(1, starters_1, 2, starters_2))
        
        verdict_r1, verdict_r1_label = judge.batched_generate(df, cfg, batch_size, 'judge_r1', evaluator_fn, agent_responses=gather_all_responses(agent_1_discuss_r1, agent_2_discuss_r1, agent_3_discuss_r1))

        agent_1_discuss_r2, a1r2_label = agent_1.batched_generate(df, cfg, batch_size, f'agent1_R{2}', evaluator_fn, agent_responses=gather_agent_responses(2, agent_2_discuss_r1, 3, agent_3_discuss_r1))
        agent_2_discuss_r2, a2r2_label = agent_2.batched_generate(df, cfg, batch_size, f'agent2_R{2}', evaluator_fn, agent_responses=gather_agent_responses(1, agent_1_discuss_r1, 3, agent_3_discuss_r1))
        agent_3_discuss_r2, a3r2_label = agent_3.batched_generate(df, cfg, batch_size, f'agent3_R{2}', evaluator_fn, agent_responses=gather_agent_responses(1, agent_1_discuss_r1, 2, agent_2_discuss_r1))

        verdict_r2, verdict_r2_label = judge.batched_generate(df, cfg, batch_size, 'judge_r2', evaluator_fn, agent_responses=gather_all_responses(agent_1_discuss_r2, agent_2_discuss_r2, agent_3_discuss_r2))

        agent_1_discuss_r3, a1r3_label = agent_1.batched_generate(df, cfg, batch_size, f'agent1_R{3}', evaluator_fn, agent_responses=gather_agent_responses(2, agent_2_discuss_r2, 3, agent_3_discuss_r2))
        agent_2_discuss_r3, a2r3_label = agent_2.batched_generate(df, cfg, batch_size, f'agent2_R{3}', evaluator_fn, agent_responses=gather_agent_responses(1, agent_1_discuss_r2, 3, agent_3_discuss_r2))
        agent_3_discuss_r3, a3r3_label = agent_3.batched_generate(df, cfg, batch_size, f'agent3_R{3}', evaluator_fn, agent_responses=gather_agent_responses(1, agent_1_discuss_r2, 2, agent_2_discuss_r2))

        verdict_r3, verdict_r3_label = judge.batched_generate(df, cfg, batch_size, 'judge_r3', evaluator_fn, agent_responses=gather_all_responses(agent_1_discuss_r3, agent_2_discuss_r3, agent_3_discuss_r3))

        final_df = pd.concat([
                df,
                starters_1.as_dataframe(col_suffix='start_simple'),
                pd.DataFrame({'label_starter1': starters_1_label}),
                starters_2.as_dataframe(col_suffix='start_technical'),
                pd.DataFrame({'label_starter2': starters_2_label}),
                starters_3.as_dataframe(col_suffix='start_thinker'),
                pd.DataFrame({'label_starter3': starters_3_label}),

                agent_1_discuss_r1.as_dataframe(col_suffix='agent1_r1'),
                pd.DataFrame({'label_agent1_R1': a1r1_label}),
                agent_2_discuss_r1.as_dataframe(col_suffix='agent2_r1'),
                pd.DataFrame({'label_agent2_R1': a2r1_label}),
                agent_3_discuss_r1.as_dataframe(col_suffix='agent3_r1'),
                pd.DataFrame({'label_agent3_R1': a3r1_label}),

                verdict_r1.as_dataframe(col_suffix='judge_r1'),
                pd.DataFrame({'label_judge_r1': verdict_r1_label}),

                agent_1_discuss_r2.as_dataframe(col_suffix='agent1_r2'),
                pd.DataFrame({'label_agent1_R2': a1r2_label}),
                agent_2_discuss_r2.as_dataframe(col_suffix='agent2_r2'),
                pd.DataFrame({'label_agent2_R2': a2r2_label}),
                agent_3_discuss_r2.as_dataframe(col_suffix='agent3_r2'),
                pd.DataFrame({'label_agent3_R2': a3r2_label}),

                verdict_r2.as_dataframe(col_suffix='judge_r2'),
                pd.DataFrame({'label_judge_r2': verdict_r2_label}),

                agent_1_discuss_r3.as_dataframe(col_suffix='agent1_r3'),
                pd.DataFrame({'label_agent1_R3': a1r3_label}),
                agent_2_discuss_r3.as_dataframe(col_suffix='agent2_r3'),
                pd.DataFrame({'label_agent2_R3': a2r3_label}),
                agent_3_discuss_r3.as_dataframe(col_suffix='agent3_r3'),
                pd.DataFrame({'label_agent3_R3': a3r3_label}),

                verdict_r3.as_dataframe(col_suffix='judge_r3'),
                pd.DataFrame({'label_judge_r3': verdict_r3_label}),
            ], 
            axis=1,
        )
        final_df.to_json(output_path/f"df_{savename}_final.json", orient='records')
        print('finished discussion')
        return final_df