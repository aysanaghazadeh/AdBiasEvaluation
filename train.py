from configs.training_config import get_args
from training.ppo_persuasion import train as ppo_train

if __name__ == "__main__":
    training_map = {
        "PPO": ppo_train
    }
    args = get_args()
    training_map[args.training_type](args)
