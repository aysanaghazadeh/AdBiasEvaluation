# import os
# print(os.getcwd())
# from configs.training_config import get_args
# from training.ppo_persuasion import train as ppo_train
# from training.DDPO_persuasion import train as ddpo_train

# if __name__ == "__main__":
#     training_map = {
#         "PPO": ppo_train,
#         "DDPO": ddpo_train
#     }
#     args = get_args()
#     training_map[args.training_type](args)


from Training.train_modifiedSD3 import train
from configs.training_config import get_args



def main():
    args = get_args()
    # trainer = Trainer(args)
    train(args)


if __name__ == "__main__":
    
    main()
