from typing import Literal, Callable
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams
from sqlgen.base_agent import TextToSQL
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
            f"You are a helpful SQL coding assistant{' ' + persona if persona else ''}. Can you please generate a SQLite query to "
            "answer the given question, based on the schema below? In your response, first briefly "
            "explain your reasoning. Your final answer should be enclosed in a markdown code block.\n\n"
            f"### Question:\n{question}\n\n"
            f"### Schema:\n{schema}\n\n"
            f"### Response:\n"
        )
        return prompt

class DiscussionAgent(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple[str, str, dict[int, str]]:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        other_responses = ''.join(
            f"### Agent {agent} Response:\n{response}'\n\n'"
            for agent, response in agent_responses.items()
        )
        prompt = (
            "You are a helpful SQL coding agent. You understand that collaborative discussion is the best way to solve problems. "
            "Using the other agent's response as additional information, your job is to generate a SQLite query to answer a "
            "given question based on the schema. In your response, please explain your reasoning clearly so that others may "
            "give you constructive feedback. Your final answer should be enclosed in a markdown code block.\n\n"
            f"### Question:\n{question}\n\n"
            f"### Schema:\n{schema}\n\n"
            f"{other_responses}"
            f"### Your Response:\n"
        )
        return prompt
    
class DiscussionJudge(TextToSQL):
    def process_bird_df(self, idx: int, row: pd.DataFrame, agent_responses: list[dict[int, str]]) -> tuple:
        schema, question = super().process_bird_df(idx, row)
        return schema, question, agent_responses[idx]
    
    def generate_prompt(self, schema: str, question: str, agent_responses: dict[int, str]) -> str:
        n_agents = len(agent_responses)
        other_responses = ''.join(
            f"### Agent {agent} Response:\n{response}'\n\n'"
            for agent, response in agent_responses.items()
        )
        prompt = (
            f"You are a SQL expert overseeing {n_agents} coding agents collaborating to answer the given question. "
            "Using the other agents' responses as additional information, generate the production-ready SQLite query. "
            "Your final answer should be enclosed in a markdown code block.\n\n"
            f"### Question:\n{question}\n\n"
            f"### Schema:\n{schema}\n\n"
            f"{other_responses}"
            f"### Your Response:\n"
        )
        return prompt



def discuss(
    df: pd.DataFrame, databases: dict[str, SQLiteDatabase], llm: LLM,
    output_path: Path, savename: str, batch_size: int, 
    evaluator_fn: Callable, db_exec_timeout: int,
) -> list[str]:
                
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
        
        def gather_agent_responses(agent_num1: int, responses_1: list[str], agent_num2: int, responses_2: list[str]) -> list[dict[int, str]]:
            agent_responses = [
                {agent_num1: resp1, agent_num2: resp2}
                for resp1, resp2 in zip(responses_1, responses_2)
            ]
            return agent_responses
        
        starters_1 = starter.batched_generate(df, cfg, batch_size, 'starter1', evaluator_fn, db_exec_timeout, persona='simple')
        starters_2 = starter.batched_generate(df, cfg, batch_size, 'starter2', evaluator_fn, db_exec_timeout, persona='technical')
        starters_3 = starter.batched_generate(df, cfg, batch_size, 'starter3', evaluator_fn, db_exec_timeout, persona='thinker')

        agent_1_discuss_r1 = agent_1.batched_generate(df, cfg, batch_size, f'agent_1_R{1}', evaluator_fn, db_exec_timeout, agent_responses=gather_agent_responses(2, starters_2.raw_responses, 3, starters_3.raw_responses))
        agent_2_discuss_r1 = agent_2.batched_generate(df, cfg, batch_size, f'agent_2_R{1}', evaluator_fn, db_exec_timeout, agent_responses=gather_agent_responses(1, starters_1.raw_responses, 3, starters_3.raw_responses))
        agent_3_discuss_r1 = agent_3.batched_generate(df, cfg, batch_size, f'agent_3_R{1}', evaluator_fn, db_exec_timeout, agent_responses=gather_agent_responses(1, starters_1.raw_responses, 2, starters_2.raw_responses))
        
        agent_1_discuss_r2 = agent_1.batched_generate(df, cfg, batch_size, f'agent_1_R{2}', evaluator_fn, db_exec_timeout, agent_responses=gather_agent_responses(2, agent_2_discuss_r1.raw_responses, 3, agent_3_discuss_r1.raw_responses))
        agent_2_discuss_r2 = agent_2.batched_generate(df, cfg, batch_size, f'agent_2_R{2}', evaluator_fn, db_exec_timeout, agent_responses=gather_agent_responses(1, agent_1_discuss_r1.raw_responses, 3, agent_3_discuss_r1.raw_responses))
        agent_3_discuss_r2 = agent_3.batched_generate(df, cfg, batch_size, f'agent_3_R{2}', evaluator_fn, db_exec_timeout, agent_responses=gather_agent_responses(1, agent_1_discuss_r1.raw_responses, 2, agent_2_discuss_r1.raw_responses))

        def gather_all_responses(responses_1: list[str], responses_2: list[str], responses_3: list[str]) -> list[dict[int, str]]:
            agent_responses = [
                {1: resp1, 2: resp2, 3: resp3}
                for resp1, resp2, resp3 in zip(responses_1, responses_2, responses_3)
            ]
            return agent_responses
        
        verdict = judge.batched_generate(df, cfg, batch_size, 'judge', evaluator_fn, db_exec_timeout, agent_responses=gather_all_responses(agent_1_discuss_r2.raw_responses, agent_2_discuss_r2.raw_responses, agent_3_discuss_r2.raw_responses))

        final_df = pd.concat([
                df, 
                starters_1.as_dataframe(col_prefix='start_simple'),
                starters_2.as_dataframe(col_prefix='start_technical'),
                starters_3.as_dataframe(col_prefix='start_thinker'),
                agent_1_discuss_r1.as_dataframe(col_prefix='agent_1_r1'),
                agent_2_discuss_r1.as_dataframe(col_prefix='agent_2_r1'),
                agent_3_discuss_r1.as_dataframe(col_prefix='agent_3_r1'),
                agent_1_discuss_r2.as_dataframe(col_prefix='agent_1_r2'),
                agent_2_discuss_r2.as_dataframe(col_prefix='agent_2_r2'),
                agent_3_discuss_r2.as_dataframe(col_prefix='agent_3_r3'),
                verdict.as_dataframe(col_prefix='judge')
            ], 
            axis=1,
        )
        final_df.to_json(output_path/f"df_{savename}_final.json", orient='records')

        return final_df