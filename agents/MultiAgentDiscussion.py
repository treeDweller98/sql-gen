import pandas as pd
from tqdm import tqdm
from agents.TextToSQL import TextToSQL
from core.Model import GenerationConfig

class MultiAgentDiscussion(TextToSQL):
    def generate_zeroshot_starter(self, schema: str, question: str, use_meta_prompt: bool, cfg: GenerationConfig) -> str:
        """ TODO: add more prompt options for more agents.
        Generates the starter responses for multi-agent discussion using either zero-shot or meta-prompt.
        Different starting prompts improve performance: https://arxiv.org/pdf/2305.14325
        """
        if use_meta_prompt:
            system_prompt = (
                "You are an SQLite expert who excels at writing queries based on the schema below. "
                "Your job is to write a valid SQLite query to answer a given user question."
                "Here is how you should approach the problem:\n"
                "1. Begin your response with 'Let\'s think step by step.'\n"
                "2. Analyze the question and schema carefully, showing all your workings:\n"
                "   - Decompose the question into subproblems.\n"
                "   - Identify the tables and the columns required to write the query.\n"
                "   - Identify the operations you will need to perform.\n"
                "3. Review your choices before generation:\n"
                "   - Identify if you missed any tables and columns.\n"
                "   - Identify if you picked any unnecessary tables and columns.\n"
                "   - Identify any unnecessary subqueries, joins, groupings, orderings, sortings etc.\n"
                "4. Ensure your choices are the correct and optimal ones.\n"
                "5. Finally, show your reasoning and write down the SQLite query.\n\n"
                f"### Schema:\n{schema}"
            )
            user_prompt = (
                f"### Question:\n{question}."
            )
        else:
            system_prompt = (
                f"Given the following SQLite tables, your job is to write SQLite queries given a user’s request. "
                f"Please begin your response with 'Let\'s think step by step'\n\n{schema}"
            )
            user_prompt   = question

        reply = self.llm(
            messages=[
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ],
            cfg=cfg
        )
        return reply
    
    def generate_discussion(self, schema: str, question: str, hint: str, agent_responses: dict[str, str], cfg: GenerationConfig) -> str:
        """ Generates a round of discussion for an agent, taking other agents' responses into account. """
        system_prompt = ( 
            "You are an SQLite expert who excels at writing, critiquing, and correcting SQL queries. "
            "Agents are discussing what would be the correct SQL query to answer the question below, based " 
            "on the database schema given. Your task is to use the other agents' response as additional " 
            "information to formulate your own solution to the problem. Please update and respond to "
            "the other agents. Address their shortcomings and any inconsistencies in their reasoning. "
            "Your final answer should be a single, valid SQLite query.\n\n"
            f"### Schema:\n{schema}\n\n"
            f"### Question:\n{question} Hint: {hint}."
        )
        user_prompt = '\n\n'.join(
            f"### {agent} Response\n{response}"
            for agent, response in agent_responses.items()
        )
        reply = self.llm(
            messages=[
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ],
            cfg=cfg
        )
        return reply
    
    def generate_verdict(self, schema: str, question: str, hint: str, agent_responses: dict[str, str], cfg: GenerationConfig) -> str:
        """ Takes the responses of agents and delivers a verdict by choosing the correct response """
        system_prompt = (
            "You are a judge. You will be given responses from agents discussing how to generate "
            "a SQLite query to answer a question, based on the database schema given below. Examine "
            "their responses carefully and select the correct query. In your final answer, please "
            "explain your reasoning and then output the correct SQL query.\n\n"
            f"### Schema:\n{schema}\n\n"
            f"### Question:\n{question} Hint: {hint}."
        )
        user_prompt = '\n\n'.join(
            f"### {agent} Response\n{response}"
            for agent, response in agent_responses.items()
        )
        reply = self.llm(
            messages=[
                { "role": "system", "content": system_prompt }, 
                { "role": "user",   "content": user_prompt },
            ],
            cfg=cfg
        )
        return reply

    ## INCORPORATE GenerationConfig first
    def generate_response(self, schema: str, question: str, hint: str, rounds: int = 3) -> dict[str, str]:
        raise NotImplementedError
        agent_1 = [self.generate_zeroshot_starter(schema, question, hint, use_meta_prompt=True)]
        agent_2 = [self.generate_zeroshot_starter(schema, question, hint, use_meta_prompt=False)]

        # TODO: extend for N agents
        for i in range(rounds):
            responses_for_agent_1 = {"Agent_2": agent_2[-1]}
            responses_for_agent_2 = {"Agent_1": agent_1[-1]}
            agent_1.append(self.generate_discussion(schema, question, hint, responses_for_agent_1))
            agent_2.append(self.generate_discussion(schema, question, hint, responses_for_agent_2))

        responses_for_judge = {"Agent_1": agent_1[-1], "Agent_2": agent_2[-1]}
        verdict = self.generate_verdict(schema, question, hint, responses_for_judge)

        return {'agent_meta_prompt': agent_1, 'agent_zero_shot': agent_2, 'verdict': verdict}
    
    def batched_generate(self, df: pd.DataFrame, rounds: int = 3) -> list[str]:
        raise NotImplementedError
        assert {'db_id', 'question', 'evidence'}.issubset(df.columns), \
        "Ensure {'db_id', 'question', 'evidence'} in df.columns"
        
        raw_responses: list[dict[str, str]] = []
        for i, row in tqdm(df.iterrows(), desc=f'{self.__agent_name} Generating SQL', total=len(df)):
            schema   = self.db_schemas[row['db_id']]
            question = row['question']
            hint     = row['evidence']
            try:
                reply = self.generate_response(schema, question, hint, rounds)
                raw_responses.append(reply)
            except Exception as e:
                self.dump_to_json_on_error(raw_responses)
                raise e
            
        return raw_responses