import os
import numpy as np
import torch
import myenv
import json


def main():
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--task', choices=['horizontal', 'vertical', 'upper'], default='horizontal', help='Task mode parameter')
    parser.add_argument('--arm', choices=['normal', 'short', 'long'], default='normal', help='Arm length parameter')
    parser.add_argument('--k1', type=float, default=0.002, help='Coefficient of torque penalty')
    parser.add_argument('--k2', type=float, default=0.1, help='Coefficient of velocity penalty')
    parser.add_argument('--algo', choices=['ppo', 'sac'], default='ppo', help='RL algorithm')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id to use; CPU is used if -1')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=10000, help='Number of steps of interaction in an epoch')
    parser.add_argument('--hidden-units', type=int, nargs='+', default=(128, 64, 32, 16))
    parser.add_argument('--activation', choices=['sigmoid', 'tanh', 'relu'], default='relu')
    parser.add_argument('--start-steps', type=int, default=10000, help="Only in SAC")
    parser.add_argument('--update-after', type=int, default=1000, help="Only in SAC")
    parser.add_argument('--update-every', type=int, default=50, help="Only in SAC")
    parser.add_argument('--alpha', type=float, default=0.2, help="Only in SAC")
    parser.add_argument('--lr', type=float, default=0.001, help="Currently only in SAC")
    parser.add_argument('--out', type=str, default='results/result', help='Output directory')
    args = parser.parse_args()

    # Create the output directory
    os.makedirs(args.out, exist_ok=True)

    # Save commandline arguments
    with open(os.path.join(args.out, "arg_params.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    # A function to make an environment
    def make_env():
        if args.arm == "normal":
            arm_type = 0
        if args.arm == "short":
            arm_type = 1
        if args.arm == "long":
            arm_type = 2

        env = myenv.ArmReachingFixedPoints(arm_type=arm_type)

        # Set task settings
        if args.task == "horizontal":
            env.target_mode = 0
        elif args.task == "vertical":
            env.target_mode = 1
        elif args.task == "upper":
            env.target_mode = 2

        # Set reward coefficients
        env.alpha1 = args.k1
        env.alpha2 = args.k2

        return env

    # Enable GPU if specified
    if args.gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Setup NN parameters
    model_params = {
        'hidden_sizes': args.hidden_units
    }
    if args.activation == 'sigmoid':
        model_params['activation'] = torch.nn.Sigmoid
    elif args.activation == 'tanh':
        model_params['activation'] = torch.nn.Tanh
    elif args.activation == 'relu':
        model_params['activation'] = torch.nn.ReLU

    # Setup RL parameters
    rl_params = {
        'env_fn': make_env,
        'ac_kwargs': model_params,
        'steps_per_epoch': args.steps_per_epoch,
        'epochs': args.epochs,
        'max_ep_len': int(1e5),
        'logger_kwargs': dict(output_dir=args.out)
    }

    if args.algo == 'ppo':
        #from spinup import ppo_pytorch as ppo
        from algos_gpu.ppo_gpu import ppo

        # Use GPU version if specified
        if args.gpu >= 0:
            from algos_gpu import ppo_core_gpu
            rl_params['actor_critic'] = ppo_core_gpu.MLPActorCritic

        # Run PPO
        ppo(**rl_params)

    if args.algo == 'sac':
        from spinup import sac_pytorch as sac

        # Add SAC-specific parameters
        rl_params['alpha'] = 0.2
        rl_params['start_steps'] = args.start_steps
        rl_params['update_after'] = args.update_after
        rl_params['update_every'] = args.update_every
        rl_params['lr'] = args.lr
        rl_params['alpha'] = args.alpha

        # Use GPU version if specified
        if args.gpu >= 0:
            print("sac_gpu")
            from algos_gpu import sac_gpu, sac_core_gpu
            rl_params['actor_critic'] = sac_core_gpu.MLPActorCritic
            sac = sac_gpu.sac  # Overwrite sac with sac_gpu.sac

        # Run SAC
        sac(**rl_params)


if __name__ == '__main__':
    main()
