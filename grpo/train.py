import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from core.dbhandler import SQLiteDatabase
from grpo.data import make_bird_grpo_dataset
from grpo.reward_utils import set_db
from grpo.reward import (
    correctness_reward_func,
    valid_sql_reward_func,
    table_col_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    tags_count_reward_func,
)

# TODO: Setup wandb, remove torch import, figure out device map, add GRPOConfig.use_vllm=True
def train_grpo(args, df: pd.DataFrame, databases: dict[str, SQLiteDatabase]):

    dataset = make_bird_grpo_dataset(df, databases)
    set_db(databases)

    model = AutoModelForCausalLM.from_pretrained(
        args.MODEL.value,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None             # should it be none?
    ).to("cuda")                    # using multi-gpu --- verify

    tokenizer = AutoTokenizer.from_pretrained(args.MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['<think>', '</think>'])
    model.resize_token_embeddings(len(tokenizer))

    training_args = GRPOConfig(
        output_dir=args.OUTPUT_PATH,
        run_name=f'{args.MODEL}_{args.EXPERIMENT}',
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,      # TODO
        num_generations=16,
        max_prompt_length=2048,             # Verify 
        max_completion_length=4096,
        num_train_epochs=1,                 # set to whatever
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",                  # setup wandb
        log_on_each_node=False,

        use_vllm=True,                      # figure out if server needs to be served
        vllm_device="auto",
        vllm_gpu_memory_utilization=0.9,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            valid_sql_reward_func,
            table_col_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            tags_count_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()