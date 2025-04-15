from copy import copy
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from Evaluation.persuasion import Persuasion
from util.data.data_util import get_train_LLAMA3_PPO_Dataset
import os
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import wandb
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
                                                                    peft_config=lora_config,
                                                                    load_in_4bit=True,
                                                                )
    # ppo_model.gradient_checkpointing_enable()
    # ppo_model = prepare_model_for_kbit_training(ppo_model)
    # ppo_model = get_peft_model(ppo_model, lora_config).to('cuda:0')
    return ppo_model, tokenizer

def train(args): # Example
    ppo_model, tokenizer = get_model(args)
    reward_model = Persuasion(args)
    dataset = get_train_LLAMA3_PPO_Dataset(args)
    
    # Set up generation configuration
    # generation_config = GenerationConfig(
    #     pad_token_id=tokenizer.eos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     do_sample=True,
    #     max_new_tokens=50,
    #     top_p=1.0,
    #     top_k=0.0,
    #     min_length=1
    # )
    # # Add generation_config to the model
    # ppo_model.generation_config = generation_config
    
    # Create a separate value model
    # value_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # value_model = prepare_model_for_kbit_training(value_model)
    # value_model = get_peft_model(value_model, LoraConfig(
    #     r=16,
    #     lora_alpha=64,
    #     lora_dropout=0.05,
    #     bias="none",
    #     peft_type=TaskType.CAUSAL_LM,
    # ))
    
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        # log_with="wandb",
        # project_kwargs={"project": "ppo-persuasion"},
        # logging_steps=10,  # Log every 10 PPO updates
        output_dir=os.path.join(args.model_path, f"ppo_{args.model_name}_{args.evaluation_model}")
    )
    wandb.init(project="llama3-ppo-chat")
    def data_collator(data):
        prompts = [item["prompt"] for item in data]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        return {"input_ids": tokenized["input_ids"]}

    # Initialize trainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=ppo_model,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        reward_model=reward_model,
        ref_model=None,  # Set ref_model to a copy of the policy model
        # value_model=value_model  # Explicitly set the value model
    )
    for epoch in range(args.epoch):
        for i, batch in enumerate(ppo_trainer.dataloader):
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
            outputs = [reward_model(r) for r in responses]
            rewards = [o[0] for o in outputs]
            images = [o[1] for o in outputs]
            # Run PPO step
            stats = ppo_trainer.step(queries, responses, rewards)
            ppo_trainer.logger.log(
                {
                    "custom_metric/mean_reward": sum(rewards) / len(rewards),
                    "num_tokens": sum(len(r.split()) for r in responses),
                }
            )
            if i % 10 == 0:
                wandb.log(stats)
                wandb.log({f"sample_{i}": wandb.Image(images[0], caption=f"Prompt: {queries[0]}")})
                print(f'epoch: {epoch}, step : {i}, \n queries: {queries}, \nresponses: {responses}')
                print(f'stats: {stats}')
                print('-'*100)