from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
from Evaluation.persuasion import Persuasion
from util.data.data_util import get_train_LLAMA3_PPO_Dataset
import os
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# Convert to value-head model

def get_model(args):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        peft_type=TaskType.CAUSAL_LM,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                                                                    args.model_name,
                                                                )
    ppo_model.gradient_checkpointing_enable()
    ppo_model = prepare_model_for_kbit_training(ppo_model)
    ppo_model = get_peft_model(ppo_model, lora_config).to('cuda:0')
    return ppo_model, tokenizer

def train(args): # Example
    ppo_model, tokenizer = get_model(args)
    reward_model = Persuasion(args)
    dataset = get_train_LLAMA3_PPO_Dataset(args)
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        log_with="wandb",
        project_kwargs={"project": "ppo-persuasion"},
        logging_steps=10,  # Log every 10 PPO updates
        output_dir=os.path.join(args.model_path, f"ppo_{args.model_name}_{args.evaluation_model}")
    )

    # Initialize trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: {"input_ids": tokenizer(data["prompt"], return_tensors="pt", padding=True)["input_ids"]}
    )

    for batch in ppo_trainer.dataloader:
        queries = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        responses = []
        generation_kwargs = {
                                "min_length": 1,
                                "top_k": 0.0,
                                "top_p": 1.0,
                                "max_new_tokens": 50,
                                "do_sample": True,
                                "pad_token_id": tokenizer.eos_token_id,
                            }
        for query in queries:
            
            response = ppo_trainer.generate(query, **generation_kwargs)
            responses.append(response)

        # Get rewards using your custom function
        rewards = [reward_model(r) for r in responses]

        # Run PPO step
        stats = ppo_trainer.step(queries, responses, rewards)
        ppo_trainer.logger.log(
            {
                "custom_metric/mean_reward": sum(rewards) / len(rewards),
                "num_tokens": sum(len(r.split()) for r in responses),
            }
        )