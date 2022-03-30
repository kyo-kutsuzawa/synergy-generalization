import os
import numpy as np
import torch
import pickle
import myenv


def main():
    import argparse
    import json

    # Process commandline arguments
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument('--model', type=str, default=None, help='File name of a trained model')
    parser.add_argument('--task', choices=['as-is', 'horizontal', 'vertical', 'facing', 'upper'], default='as-is', help='Type of task. When specified as-is, the same task as the training is used.')
    parser.add_argument('--n-rollouts', type=int, default=8, help='Number of trajectories to be recorded')
    parser.add_argument('--norender', action='store_false', dest='render', help='')
    parser.add_argument('--out', type=str, default=None, help='File name where results to be saved. If not specified, {working dirctory}/{model name}.pickle is used')
    args = parser.parse_args()

    # Load a policy
    model = torch.load(args.model, map_location=torch.device('cpu'))

    # Load training settings
    _filename_training_args = os.path.join(os.path.dirname(args.model), "../arg_params.json")
    with open(_filename_training_args, mode="r") as f:
        training_args = json.load(f)

    if args.task == "as-is":
        task = training_args["task"]
    else:
        task = args.task

    # Setup record variables
    data = {
        "t": [],
        "observation": [],
        "action": [],
        "finger_pos": [],
        "target reward": [],
        "vel reward": [],
        "energy reward": []
    }

    if training_args["arm"] == "normal":
        arm_type = 0
    if training_args["arm"] == "short":
        arm_type = 1
    if training_args["arm"] == "long":
        arm_type = 2

    # Create an environment
    env = myenv.ArmReachingDeterministic(target_position=np.zeros(3), arm_type=arm_type)
    env.reset()

    # Perform rollouts
    for i in range(args.n_rollouts):
        # Setup the target position
        if task == "horizontal":
            r = 0.15  # m
            theta = 2*np.pi * i / args.n_rollouts
            target_position = env.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])
        elif task == "vertical":
            r = 0.15  # m
            theta = 2*np.pi * i / args.n_rollouts
            target_position = env.target_center + r * np.array([-np.sin(theta), 0.0, -np.cos(theta)])
        elif task == "facing":
            r = 0.15  # m
            theta = 2*np.pi * i / args.n_rollouts
            target_position = env.target_center + r * np.array([0.0, -np.sin(theta), -np.cos(theta)])
        elif task == "upper":
            r = 0.15  # m
            theta = 2*np.pi * i / args.n_rollouts
            target_position = env.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])
            target_position += np.array([0.0, 0.0, 0.05])
        env.set_target_position(target_position)

        # Add new sub-lists to the record data
        for k in data.keys():
            data[k].append([])

        # perform a rollout
        obs = env.reset()
        done = False
        while not done:
            # Render the simulation, if specified
            if args.render:
                env.render()
            else:
                env.record()

            # Compute an action
            with torch.no_grad():
                obs = torch.as_tensor(obs, dtype=torch.float32)
                pi = model.pi._distribution(obs)
                action = pi.mean

            # Record the current states
            data["t"][-1].append(env.t)
            data["observation"][-1].append(obs.tolist())
            data["action"][-1].append(action.tolist())
            data["finger_pos"][-1].append(env.fingertip_position.tolist())

            # Step the simulation
            obs, _, done, info = env.step(action)

            # Record rewards
            data["target reward"][-1].append(info["tracking reward"])
            data["vel reward"][-1].append(info["velocity reward"])
            data["energy reward"][-1].append(info["energy reward"])

        if not args.render:
            env.save_video("result/video{}.mp4".format(i))

    # Convert the record to ndarray
    for k, v in data.items():
        data[k] = np.array(v)

    # Record task information
    data["task"] = task

    # Save the record as a pickle file
    if args.out is None:
        dirname = os.path.dirname(os.path.dirname(args.model))
        args.out = os.path.basename(dirname) + ".pickle"
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
