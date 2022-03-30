import os
import numpy as np
import pickle
import myenv
import synergy


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=8, help='Number of trajectories to optimize')
    parser.add_argument('--task', choices=['horizontal', 'vertical', 'facing', 'upper'], default='horizontal', help='Task parameter')
    parser.add_argument('--arm', choices=['normal', 'short', 'long'], default='normal', help='Arm length parameter')
    parser.add_argument('--n-synergies-1', type=int, default=3, help='Number of motor synergies for the horizontal policy')
    parser.add_argument('--n-synergies-2', type=int, default=3, help='Number of motor synergies for the vertical policy')
    parser.add_argument('--out', type=str, default='result', help='Output directory')
    args = parser.parse_args()

    # Setup constant values
    is_multiprocessing = True
    trial = trial_cmaes
    if args.n_synergies_1 + args.n_synergies_2 == 1:
        trial = trial_cmaes_1dim

    # Extract synergies from record files
    model = load_synergy_model(args.n_synergies_1, args.n_synergies_2, args.arm)

    # Setup environments
    env_list = []
    for i in range(args.n_trials):
        env = make_env(i, args.n_trials, args.task)
        env_list.append(env)

    # Optimize synergy activities
    if is_multiprocessing:
        import multiprocessing
        trials_args = [(i, env_list[i], model) for i in range(args.n_trials)]
        p = multiprocessing.Pool(os.cpu_count())
        results = p.starmap(trial, trials_args)
        p.close()
    else:
        results = []
        for i in range(args.n_trials):
            result = trial(i, env_list[i], model)
            results.append(result)

    # Setup record variables
    data = {
        "t": [],
        "observation": [],
        "action": [],
        "activity": [],
        "finger_pos": [],
        "target reward": [],
        "vel reward": [],
        "energy reward": []
    }

    # Evaluate the optimal synergy activities
    for i in range(args.n_trials):
        # Add new sub-lists to the record data
        for k in data.keys():
            data[k].append([])

        env = env_list[i]
        activities = results[i]["final"]
        actions = model.decode(activities)  # shape: (1, T, n_act)
        data["activity"][-1].extend(activities.flatten())

        obs = env.reset()
        done = False
        while not done:
            # Compute an action
            action = actions[0, env.cnt, :]

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

        data["t"][-1].append(env.t)
        data["observation"][-1].append(obs.tolist())
        data["finger_pos"][-1].append(env.fingertip_position.tolist())

    # Convert the record to ndarray
    for k, v in data.items():
        data[k] = np.array(v)

    # Record other information
    data["task"] = args.task
    data["model"] = model

    # Save optimization results
    os.makedirs(args.out, exist_ok=True)
    filename = "optimization_results.pickle"
    with open(os.path.join(args.out, filename), 'wb') as f:
        pickle.dump(data, f)

    # Save intermediate results
    intermediate_results = {
        "activities": results,
        "task": args.task,
        "model": model,
        "envs": env_list,
    }
    filename = "intermediate_results.pickle"
    with open(os.path.join(args.out, filename), 'wb') as f:
        pickle.dump(intermediate_results, f)


def make_env(idx, n_directions, task):
    # Make a deterministic environment
    env = myenv.ArmReachingDeterministic()

    # Setup initial condition
    qpos0 = env.init_qpos
    qvel0 = env.init_qvel
    env.set_initial_states(qpos0, qvel0)

    # Setup the target position
    r = 0.15  # m
    theta = 2*np.pi * idx / n_directions

    if task == "horizontal":
        target_position = env.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])
    elif task == "vertical":
        target_position = env.target_center + r * np.array([-np.sin(theta), 0.0, -np.cos(theta)])
    elif task == "facing":
        target_position = env.target_center + r * np.array([0.0, -np.sin(theta), -np.cos(theta)])
    elif task == "upper":
        target_position = env.target_center + r * np.array([-np.sin(theta), -np.cos(theta), 0.0])
        target_position += np.array([0.0, 0.0, 0.05])
    env.set_target_position(target_position)

    return env


def trial_cmaes(idx_trial, env, model):
    """Optimize synergy activities with CMA-ES (cmaes module)
    """
    import cmaes
    n_iteration = 500
    results = {}

    # Initialize random seed
    np.random.seed(idx_trial * np.random.randint(1, 100))

    # Optimize synergy activities
    x_init = np.full(model.n_synergies, 0.0)
    sigma0 = 0.001
    optimizer = cmaes.CMA(mean=x_init, sigma=sigma0)
    print("CMA-ES (dim={}, popsize={})".format(optimizer.dim, optimizer.population_size))
    for generation in range(n_iteration):
        solutions = []
        reward_sum = 0.0
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = objective_func(x, env, model)
            solutions.append((x, value))
            reward_sum += value
        optimizer.tell(solutions)
        if generation % 10 == 0:
            print("Worker {} #{}: R={}".format(idx_trial, generation, reward_sum/optimizer.population_size))
            results[generation] = actvation(optimizer._mean).reshape((1, model.n_synergies))
    x_opt = optimizer._mean
    activities = actvation(x_opt).reshape((1, model.n_synergies))
    results["final"] = activities

    return results


def trial_cmaes_1dim(idx_trial, env, model):
    """Optimize synergy activities with CMA-ES (cmaes module)
    """
    import cmaes
    n_iteration = 500
    results = {}

    # Initialize random seed
    np.random.seed(idx_trial * np.random.randint(1, 100))

    # Optimize synergy activities
    x_init = np.full(model.n_synergies+1, 0.0)
    sigma0 = 0.001
    optimizer = cmaes.CMA(mean=x_init, sigma=sigma0)
    print("CMA-ES (dim={}, popsize={})".format(optimizer.dim, optimizer.population_size))
    for generation in range(n_iteration):
        solutions = []
        reward_sum = 0.0
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = objective_func(x[0], env, model)
            solutions.append((x, value))
            reward_sum += value
        optimizer.tell(solutions)
        if generation % 10 == 0:
            print("Worker {} #{}: R={}".format(idx_trial, generation, reward_sum/optimizer.population_size))
            results[generation] = actvation(optimizer._mean[0]).reshape((1, model.n_synergies))
    x_opt = optimizer._mean[0]
    activities = actvation(x_opt).reshape((1, model.n_synergies))
    results["final"] = activities

    return results


def load_synergy_model(n_synergies_1, n_synergies_2, arm):
    if n_synergies_1 > 0:
        with open("data/result_h_{}.pickle".format(arm), 'rb') as f:
            data = pickle.load(f)
        synergy_model1 = synergy.SpatioTemporalSynergy(n_synergies=n_synergies_1, method="negative-nmf")
        synergy_model1.extract(data['action'])

    if n_synergies_2 > 0:
        with open("data/result_s_{}.pickle".format(arm), 'rb') as f:
            data = pickle.load(f)
        synergy_model2 = synergy.SpatioTemporalSynergy(n_synergies=n_synergies_2, method="negative-nmf")
        synergy_model2.extract(data['action'])

    if n_synergies_1 > 0 and n_synergies_2 > 0:
        synergy_model = synergy.CombinedSpatioTemporalSynergy(synergy_model1, synergy_model2)
    elif n_synergies_1 > 0:
        synergy_model = synergy_model1
    elif n_synergies_2 > 0:
        synergy_model = synergy_model2

    return synergy_model


def objective_func(x, env, model):
    """Evaluate synergy activities.
    """
    # Decode actions
    x = actvation(x)
    activities = x.reshape((1, model.n_synergies))
    X = model.decode(activities)  # shape: (1, T, n_act)

    # Execute a trial
    env.alpha1 = 0.002
    env.alpha2 = 0.2
    env.reset()
    r_total = 0.0
    done = False
    cnt = 0
    while not done:
        action = X[0, cnt, :]
        obs, r, done, _ = env.step(action)
        r_total += float(r)
        cnt += 1

    return -r_total


def actvation(x):
    """Map real values (-infinity to infinity) to non-negative values (0 to infinity)
    """
    return abs(x)


if __name__ == "__main__":
    main()
