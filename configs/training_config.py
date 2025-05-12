# import torch
# import yaml
# import argparse


# def read_yaml_config(file_path):
#     """Reads a YAML configuration file and returns a dictionary of the settings."""
#     with open(file_path, 'r') as file:
#         return yaml.safe_load(file)


# def convert_to_args(config):
#     """Converts a nested dictionary to a list of command-line arguments."""

#     args = {}
#     for section, settings in config.items():
#         for key, value in settings.items():
#             args[key] = value
#     return args


# def set_conf(config_file):
#     yaml_file_path = config_file
#     config = read_yaml_config(yaml_file_path)
#     args = convert_to_args(config)
#     return args


# def parse_args():
#     """ Parsing the Arguments for the Advertisement Generation Project"""
#     parser = argparse.ArgumentParser(description="Configuration arguments for advertisement generation:")
#     parser.add_argument('--config_type',
#                         type=str,
#                         required=True,
#                         help='Choose among ARGS for commandline arguments, DEFAULT for default values, or YAML for '
#                              'config file')
#     parser.add_argument('--task',
#                         type=str,
#                         default='PittAd',
#                         help='Choose between PittAd, whoops')
#     parser.add_argument('--AD_type',
#                         type=str,
#                         default='COM',
#                         choices=['COM', 'PSA', 'all'])
#     parser.add_argument('--description_goal',
#                         type=str,
#                         default='prompt_expansion',
#                         choices=['prompt_expansion', 'image_descriptor'])
#     parser.add_argument('--description_type',
#                         type=str,
#                         default='IN',
#                         help='Choose among IN, UH, combine')
#     parser.add_argument('--VLM',
#                         type=str,
#                         default='llava')
#     parser.add_argument('--VLM_prompt',
#                         type=str,
#                         default='IN_description_generation.jinja')
#     parser.add_argument('--llm_prompt',
#                          type=str,
#                          default='LLM_input.jinja',
#                          help='LLM input prompt template file name.')
#     parser.add_argument('--T2I_prompt',
#                          type=str,
#                          default='LLM.jinja',
#                          help='T2I input prompt template file name.')
#     parser.add_argument('--with_sentiment',
#                         type=bool,
#                         default=False,
#                         help='True if you want to include the ground truth sentiment in the prompt.')
#     parser.add_argument('--with_topics',
#                         type=bool,
#                         default=False,
#                         help='True if you want to include the ground truth topic in the prompt.')
#     parser.add_argument('--with_audience',
#                         type=bool,
#                         default=False,
#                         help='True if you want to include the detected audience by LLM in the prompt.')
#     parser.add_argument('--model_path',
#                         type=str,
#                         default='../models',
#                         help='The path to trained models')
#     parser.add_argument('--config_path',
#                         type=str,
#                         default=None,
#                         help='The path to the config file if config_type is YAML')
#     parser.add_argument('--result_path',
#                         type=str,
#                         default='../experiments',
#                         help='The path to the folder for saving the results')
#     parser.add_argument('--T2I_model',
#                         type=str,
#                         default='PixArt',
#                         help='T2I generation model chosen from: PixArt, Expressive, ECLIPSE, Translate')
#     parser.add_argument('--LLM',
#                         type=str,
#                         default='Mixtral7B',
#                         help='LLM chosen from: Mistral7B, phi, LLaMA3')
#     parser.add_argument('--train',
#                         type=bool,
#                         default=False,
#                         help='True if the LLM is being fine-tuned')
#     parser.add_argument('--data_path',
#                         type=str,
#                         default='../Data/PittAd',
#                         help='Path to the root of the data'
#                         )
#     parser.add_argument('--train_ratio',
#                         type=float,
#                         default=0.7,
#                         help='the ratio of the train size to the dataset in train test split.'
#                         )
#     parser.add_argument('--test_size',
#                         type=int,
#                         default=1500,
#                         help='number of example in the test-set, it must be smaller than the original test set')
#     parser.add_argument('--train_set_QA',
#                         type=str,
#                         default='train/Action_Reason_statements.json',
#                         help='If the model is fine-tuned, relative path to the train-set QA from root path')
#     parser.add_argument('--train_set_images',
#                         type=str,
#                         default='train_images_total',
#                         help='If the model is fine-tuned, relative path to the train-set Images from root path')
#     parser.add_argument('--test_set_QA',
#                         type=str,
#                         default='train/QA_Combined_Action_Reason_train.json',
#                         help='Relative path to the QA file for action-reasons from root path'
#                         )
#     parser.add_argument('--test_set_images',
#                         type=str,
#                         nargs='+',
#                         default=['train_images_all', 'train_images_all'],
#                         help='Relative path to the original images for the test set from root')
#     parser.add_argument('--text_input_type',
#                         type=str,
#                         default='LLM',
#                         help='Type of the input text for T2I generation model. Choose from LLM_generated, '
#                              'AR (for action-reason),'
#                              'original_description (for combine, VT, IN, and atypicality descriptions)')
#     parser.add_argument('--description_file',
#                         type=str,
#                         default=None,
#                         help='Path to the description file for the T2I input.')
#     parser.add_argument('--prompt_path',
#                         type=str,
#                         default='util/prompt_engineering/prompts',
#                         help='Path to the folder of prompts. Set the name of prompt files as: {text_input_type}.jinja')
#     parser.add_argument('--fine_tuned',
#                         type=bool,
#                         default=False,
#                         help='True if you want to use the fine-tuned model')
#     parser.add_argument('--api_key',
#                         type=str,
#                         default=None,
#                         help='api key for openai')
#     parser.add_argument('--evaluation_type',
#                         type=str,
#                         default='persuasion',
#                         help='evaluation type for the model')
#     parser.add_argument('--evaluation_model',
#                         type=str,
#                         default='LLAMA3_instruct',
#                         help='evaluation model for the reward model')
#     parser.add_argument('--model_name',
#                         type=str,
#                         default='meta-llama/Meta-Llama-3-8B-instruct',
#                         help='model name for the reward model')
#     parser.add_argument('--training_type',
#                         type=str,
#                         default='PPO',
#                         help='training type for the model')
#     parser.add_argument('--epoch',
#                         type=int,
#                         default=10,
#                         help='number of epochs for the training')
#     parser.add_argument('--pretrained_model',
#                         type=str,
#                         default='runwayml/stable-diffusion-v1-5',
#                         help='pretrained model for the training')
#     parser.add_argument('--pretrained_revision',
#                         type=str,
#                         default='main',
#                         help='pretrained model revision for the training')
#     parser.add_argument('--hf_hub_model_id',
#                         type=str,
#                         default='ddpo-finetuned-stable-diffusion',
#                         help='huggingface model id for the training')
#     parser.add_argument('--use_lora',
#                         type=bool,
#                         default=True,
#                         help='use lora for the training')
#     parser.add_argument('--batch_size',
#                         type=int,
#                         default=1,
#                         help='batch size for the training')
#     parser.add_argument('--output_dir',
#                         type=str,
#                         default='../models/ddpo_checkpoints',
#                         help='output directory for the training')
    
    
#     return parser.parse_args()


# def get_args():
#     args = parse_args()
#     if args.config_type == 'YAML':
#         args = set_conf(args.config_path)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     args.device = device
#     # args.device = torch.device(f'cuda:{1}')
#     print("Arguments are:\n", args, '\n', '-'*40)
#     return args





import torch
import yaml
import argparse
import os


def read_yaml_config(file_path):
    """Reads a YAML configuration file and returns a dictionary of the settings."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def convert_to_args(config):
    """Converts a nested dictionary to a list of command-line arguments."""

    args = {}
    for section, settings in config.items():
        for key, value in settings.items():
            args[key] = value
    return args


def set_conf(config_file):
    yaml_file_path = config_file
    config = read_yaml_config(yaml_file_path)
    args = convert_to_args(config)
    return args


def parse_args():
    """ Parsing the Arguments for the Advertisement Generation Project"""
    parser = argparse.ArgumentParser(description="Configuration arguments for advertisement generation:")
    #config
    parser.add_argument('--config_type',
                        type=str,
                        required=True,
                        help='Choose among ARGS for commandline arguments, DEFAULT for default values, or YAML for '
                             'config file')
    parser.add_argument('--config_path',
                        type=str,
                        default=None,
                        help='The path to the config file if config_type is YAML')
    
    #model
    parser.add_argument('--model_path',
                        type=str,
                        default='../models',
                        help='The path to trained models')
    parser.add_argument("--pretrained_model_name_or_path",
                        type=str,
                        default='stabilityai/stable-diffusion-xl-base-1.0',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    # Add these new parameters to your parser:
    parser.add_argument("--variant",
                        type=str,
                        default=None,
                        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--pretrained_vae_model_name_or_path",
                        type=str,
                        default='madebyollin/sdxl-vae-fp16-fix',
                        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",)
    parser.add_argument("--revision",
                        type=str,
                        default=None,
                        required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--train_text_encoder",
                        action="store_true",
                        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument('--T2I_model',
                        type=str,
                        default='SDXL',
                        help='T2I generation model chosen from: PixArt, Expressive, ECLIPSE, Translate, SDXL, AuraFlow, FLUX')
    parser.add_argument('--LLM',
                        type=str,
                        default='Mixtral7B',
                        help='LLM chosen from: Mixtral7B, Mistral7B, Vicuna, LLaMA2')
    parser.add_argument('--train',
                        type=bool,
                        default=True,
                        help='True if the model is being fine-tuned')
    
    #data processing
    parser.add_argument("--resolution",
                        type=int,
                        default=512,
                        help=(
                            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                            " resolution"))
    parser.add_argument("--center_crop",
                        default=False,
                        action="store_true",
                        help=(
                            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
                            " cropped. The images will be resized to the resolution first before cropping."))
    parser.add_argument("--random_flip",
                        action="store_true",
                        help="whether to randomly flip images horizontally")
    
    #validation
    parser.add_argument(
        "--validation_prompts",
        type=list,
        default=[
            'The surface of beer bottle mimics the texture of feather, while retaining its original structure.',
            'Beer glass appears to be composed of numerous, smaller instances of ice cubes, altering its texture.',
            'Cigarette is visibly located within the gun, in an unconventional manner.',
            'Pig completely replaces bride in its usual context, assuming its function or position.',
            'The surface of apple mimics the texture of kiwi, while retaining its original structure.',
            'Owl appears to be composed of numerous, smaller instances of coffee beans, altering its texture.'
        ],
        help="A list of prompts that are used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    
    #data
    parser.add_argument('--data_path',
                        type=str,
                        default='../Data/PittAd',
                        help='Path to the root of the data'
                        )
    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.4,
                        help='the ratio of the train size to the dataset in train test split.'
                        )
    parser.add_argument('--train_set_QA',
                        type=str,
                        default='QA_Combined_Action_Reason_train.json',
                        help='If the model is fine-tuned, relative path to the train-set QA from root path')
    parser.add_argument('--train_set_images',
                        type=str,
                        default='train_images_all',
                        help='If the model is fine-tuned, relative path to the train-set Images from root path')
    parser.add_argument('--test_set_QA',
                        type=str,
                        default='train/QA_Combined_Action_Reason_train.json',
                        help='Relative path to the QA file for action-reasons from root path'
                        )
    parser.add_argument('--test_set_images',
                        type=str,
                        default='train_images_all',
                        help='Relative path to the original images for the test set from root')
    parser.add_argument('--text_input_type',
                        type=str,
                        default='AR',
                        help='Type of the input text for T2I generation model. Choose from LLM_generated, '
                             'AR (for action-reason),'
                             'original_description (for combine, VT, IN, and atypicality descriptions)')
    parser.add_argument('--description_file',
                        type=str,
                        default=None,
                        help='Path to the description that includes only product name.')
    parser.add_argument('--product_file',
                        type=str,
                        default=None,
                        help='Path to the negative adjective for the action reason statements.')
    parser.add_argument('--negative_file',
                        type=str,
                        default=None,
                        help='Path to the description file for the T2I input.')
    parser.add_argument("--dataloader_num_workers",
                        type=int,
                        default=0,
                        help=(
                            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
                        ))
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=1,
                        help='Batch size (per device) for the training dataloader.')
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    #prompt
    parser.add_argument('--prompt_path',
                        type=str,
                        default='utils/prompt_engineering/prompts',
                        help='Path to the folder of prompts. Set the name of prompt files as: {text_input_type}.jinja')
    parser.add_argument('--llm_prompt',
                         type=str,
                         default='LLM_input.jinja',
                         help='LLM input prompt template file name.')
    parser.add_argument('--T2I_prompt',
                         type=str,
                         default='LLM.jinja',
                         help='T2I input prompt template file name.')
    
    #learning rate
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5,
                        help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument("--text_encoder_lr",
                        type=float,
                        default=5e-6,
                        help="Text encoder learning rate to use.")
    parser.add_argument("--scale_lr",
                        action="store_true",
                        default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler",
                        type=str,
                        default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ))
    parser.add_argument("--lr_warmup_steps", 
                        type=int, 
                        default=500, 
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles",
                        type=int,
                        default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power",
                        type=float,
                        default=1.0,
                        help="Power factor of the polynomial scheduler.")
    parser.add_argument("--sample_batch_size",
                        type=int,
                        default=1,
                        help="Batch size (per device) for sampling images.")
    parser.add_argument("--weighting_scheme",
                        type=str,
                        default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
                        help="The weighting scheme to use for training.")
    parser.add_argument("--logit_mean",
                        type=float,
                        default=0.0,
                        help="Mean to use when using the 'logit_normal' weighting scheme.")
    parser.add_argument("--logit_std",
                        type=float,
                        default=1.0,
                        help="Standard deviation to use when using the 'logit_normal' weighting scheme.")
    parser.add_argument("--mode_scale",
                        type=float,
                        default=1.29,
                        help="Scale of mode weighting scheme. Only effective when using 'mode' as the weighting_scheme.")
    parser.add_argument("--precondition_outputs",
                        type=int,
                        default=1,
                        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM.")
    parser.add_argument("--optimizer",
                        type=str,
                        default="AdamW",
                        help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    parser.add_argument("--prodigy_beta3",
                        type=float,
                        default=None,
                        help="Coefficients for computing the Prodigy stepsize using running averages. If set to None, uses the value of square root of beta2.")
    parser.add_argument("--prodigy_decouple",
                        type=bool,
                        default=True,
                        help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay_text_encoder",
                        type=float,
                        default=1e-03,
                        help="Weight decay to use for text_encoder")
    parser.add_argument("--prodigy_use_bias_correction",
                        type=bool,
                        default=True,
                        help="Turn on Adam's bias correction. True by default.")
    parser.add_argument("--prodigy_safeguard_warmup",
                        type=bool,
                        default=True,
                        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.")
    parser.add_argument("--snr_gamma",
                        type=float,
                        default=None,
                        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
                        "More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--allow_tf32",
                        action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ))
    
    #optimizer
    parser.add_argument("--use_8bit_adam", 
                        action="store_true", 
                        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prediction_type",
                        type=str,
                        default=None,
                        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    #training parameters
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help='Defines the number of epochs if train is True'
    )
    #gradient accumulation
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument('--weight_decay',
                        type=int,
                        default=0.01)
    
    #RLHF rewards
    parser.add_argument('--VLM',
                        type=str)
    parser.add_argument('--evaluation_type',
                        type=str,
                        default='image_reward')
    
    #LoRA
    parser.add_argument("--mixed_precision",
                        type=str,
                        default=None,
                        choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."))
    parser.add_argument("--lora_layers",
                        type=str,
                        default=None,
                        help=(
                            "The transformer block layers to apply LoRA training on. Please specify the layers in a comma seperated string."
                            "For examples refer to https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_SD3.md"
                        ))
    parser.add_argument("--lora_blocks",
                        type=str,
                        default=None,
                        help=(
                            "The transformer blocks to apply LoRA training on. Please specify the block numbers in a comma seperated manner."
                            'E.g. - "--lora_blocks 12,30" will result in lora training of transformer blocks 12 and 30.'
                        ))
    parser.add_argument("--upcast_before_saving",
                        action="store_true",
                        default=False,
                        help=(
                            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
                            "Defaults to precision dtype used for training to save memory"
                        ))
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", 
                        action="store_true", 
                        help="Whether or not to use xformers.")
    parser.add_argument("--noise_offset", 
                        type=float, 
                        default=0, 
                        help="The scale of noise offset.")
    parser.add_argument("--rank",
                        type=int,
                        default=4,
                        help=("The dimension of the LoRA update matrices."))
    
    #Prior preservation
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        default=False,
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss."
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU. Default to fp16 if a GPU is available else fp32."
        ),
    )
    
    #checkpointing and saving
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument('--results',
                        type=str,
                        default='../experiments',
                        help='The path to the folder for saving the results')
    
    #loggings and publishing
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id",
                        type=str,
                        default=None,
                        help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir",
                        type=str,
                        default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."))
    parser.add_argument("--report_to",
                        type=str,
                        default="wandb",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'))
    
    #reproducibility
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    parser.add_argument(
        "--enable_npu_flash_attention", 
        action="store_true", 
        help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="debug loss for each image, if filenames are available in the dataset",
    )
    
    return parser.parse_args()


def get_args():
    args = parse_args()
    if args.config_type == 'YAML':
        args = set_conf(args.config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    args.device = device
    print("Arguments are:\n", args, '\n', '-'*40)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args






