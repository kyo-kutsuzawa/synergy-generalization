# Scripts for "Motor Synergy Generalization Framework for New Targets in Multi-planar and Multi-directional Reaching Task"

## Requirements

- Python 3.8.5
    - cmaes: 0.8.2
    - gym: 0.15.7
    - matplotlib: 3.4.2
    - mujoco-py: 2.0.2.1
    - numpy: 1.20.3
    - scikit-learn: 0.24.2
    - scipy: 1.6.3
    - spinup: 0.2.0 (Available at https://github.com/openai/spinningup.git)
    - torch: 1.9.0
- MuJoCo version 2.0

## Visualize the environment

```
python src/env_test/myenv.py
```

## RL-Policy Acquisition

### Training initial policies

Horizontal policy:
```
python src/rl/train.py --out results/result_h --task horizontal --gpu 2 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
```

Sagittal policy:
```
python src/rl/train.py --out results/result_v --task vertical   --gpu 3 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
```

### Evaluation of the initial policies

Assume that there exists a folder named `result`.

Eight-points reaching in the learned targets:
```
python src/rl/evaluate.py --model results/result_h/pyt_save/model.pt --norender --out result/result_h.pickle
python src/rl/evaluate.py --model results/result_v/pyt_save/model.pt --norender --out result/result_s.pickle
```

32-points reaching in the learned planes:
```
python src/rl/evaluate.py --model results/result_h/pyt_save/model.pt --n-rollouts 32 --norender --out result/result_h-32p.pickle
python src/rl/evaluate.py --model results/result_v/pyt_save/model.pt --n-rollouts 32 --norender --out result/result_s-32p.pickle
```

Eight-points reaching in the frontal targets:
```
python src/rl/evaluate.py --model results/result_h/pyt_save/model.pt --task facing --norender --out result/result_h-frontal.pickle
python src/rl/evaluate.py --model results/result_v/pyt_save/model.pt --task facing --norender --out result/result_s-frontal.pickle
```

Eight-points reaching in the upper targets:
```
python src/rl/evaluate.py --model results/result_h/pyt_save/model.pt --task upper --norender --out result/result_h-upper.pickle
python src/rl/evaluate.py --model results/result_v/pyt_save/model.pt --task upper --norender --out result/result_s-upper.pickle
```

### Visualization of results

Trajectories of eight-points reaching in the learned targets:
```
python src/rl/show_results.py --filename result/result_h.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-horizontal.pdf

python src/rl/show_results.py --filename result/result_s.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-sagittal.pdf
```

Trajectories of 32-points reaching in the learned planes:
```
python src/rl/show_results.py --filename result/result_h-32p.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-horizontal-32p.pdf

python src/rl/show_results.py --filename result/result_s-32p.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-sagittal-32p.pdf
```

Trajectories of eight-points reaching in the frontal targets:
```
python src/rl/show_results.py --filename result/result_h-frontal.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-frontal-by-horizontal-policy.pdf

python src/rl/show_results.py --filename result/result_s-frontal.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-frontal-by-sagittal-policy.pdf
```

Trajectories of eight-points reaching in the upper targets:
```
python src/rl/show_results.py --filename result/result_h-upper.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-upper-by-horizontal-policy.pdf

python src/rl/show_results.py --filename result/result_s-upper.pickle -v pos2d err
mv result/trajectories_original.pdf result/trajectories-upper-by-sagittal-policy.pdf
```

### Get learning progress

Training:
```
python src/rl/train.py --out results/result_h_save --task horizontal --gpu 2 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
python src/rl/train.py --out results/result_v_save --task vertical   --gpu 3 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
```

Evaluation:
```
./src/rl/generate_progress.sh
```

Visualization:
```
python src/rl/show_traj_progress.py
```

## Synergy Evaluation

### R2 visualization

```
python src/synergetic/show_R2.py --filename result/result_h.pickle
mv result/R2.pdf result/R2-horizontal.pdf

python src/synergetic/show_R2.py --filename result/result_s.pickle
mv result/R2.pdf result/R2-sagittal.pdf
```

### Synergy visualization

Motor synergy waveforms:
```
python src/synergetic/show_synergy.py --filename result/result_h.pickle --n-synergies 4
mv result/motor_synergies.pdf result/motor_synergies-horizontal.pdf

python src/synergetic/show_synergy.py --filename result/result_s.pickle --n-synergies 4
mv result/motor_synergies.pdf result/motor_synergies-sagittal.pdf
```

Motor synergy activities for reaching directions:
```
python src/synergetic/show_synergy_direction.py --filename result/result_h.pickle --filename2 result/result_h-32p.pickle --n-synergies 4
mv result/synergy_direction.pdf result/synergy_direction-horizontal.pdf

python src/synergetic/show_synergy_direction.py --filename result/result_s.pickle --filename2 result/result_s-32p.pickle --n-synergies 4
mv result/synergy_direction.pdf result/synergy_direction-sagittal.pdf
```

## Trajectory Optimization with Motor Synergies

### Optimization by varying numbers of motor synergies

```
./src/synergetic/optimization_all.sh
```

### Visualization of the results

Performance maps:
```
python src/synergetic/show_performance_map.py --targets horizontal
mv result/performance_map.pdf result/performance_map-horizontal.pdf

python src/synergetic/show_performance_map.py --targets sagittal
mv result/performance_map.pdf result/performance_map-sagittal.pdf

python src/synergetic/show_performance_map.py --targets frontal
mv result/performance_map.pdf result/performance_map-frontal.pdf

python src/synergetic/show_performance_map.py --targets upper
mv result/performance_map.pdf result/performance_map-upper.pdf
```

Trajectories in the frontal targets:
```
python src/synergetic/show_results.py --filename result/optim_f1/result_5_4/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-frontal-by-optimization.pdf

python src/synergetic/show_results.py --filename result/optim_f1/result_5_0/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-frontal-by-optimization-h.pdf

python src/synergetic/show_results.py --filename result/optim_f1/result_0_8/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-frontal-by-optimization-v.pdf

python src/synergetic/show_traj_progress.py --filename result/optim_f1/result_5_4/intermediate_results.pickle
mv result/trajectory_progress.pdf result/trajectory_progress-frontal.pdf
```

Trajectories in the upper targets:
```
python src/synergetic/show_results.py --filename result/optim_u1/result_5_5/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-upper-by-optimization.pdf

python src/synergetic/show_results.py --filename result/optim_u1/result_8_0/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-upper-by-optimization-h.pdf

python src/synergetic/show_results.py --filename result/optim_u1/result_0_6/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-upper-by-optimization-v.pdf

python src/synergetic/show_traj_progress.py --filename result/optim_u1/result_5_5/intermediate_results.pickle
mv result/trajectory_progress.pdf result/trajectory_progress-upper.pdf
```

Trajectories in the horizontal targets:
```
python src/synergetic/show_results.py --filename result/optim_h1/result_5_0/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-horizontal-by-optimization.pdf
```

Trajectories in the sagittal targets:
```
python src/synergetic/show_results.py --filename result/optim_s1/result_0_5/optimization_results.pickle -v pos2d err
mv result/trajectories_synergetic.pdf result/trajectories-sagittal-by-optimization.pdf
```

Optimization progress:
```
python src/synergetic/show_traj_progress.py --filename result/optim_f1/result_5_4/optimization_results.pickle
mv result/trajectory_progress.pdf result/trajectory_progress-frontal.pdf
```

## Transfer to New Arms

### Training policies

Horizontal policy for the shorter-arm:
```
python src/rl/train.py --out results/result_h_short --task horizontal --arm short --gpu 0 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
```

Sagittal policy for the shorter-arm:
```
python src/rl/train.py --out results/result_s_short --task vertical   --arm short --gpu 0 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
```

Horizontal policy for the longer-arm:
```
python src/rl/train.py --out results/result_h_long --task horizontal  --arm long  --gpu 0 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
```

Sagittal policy for the longer-arm:
```
python src/rl/train.py --out results/result_s_long --task vertical    --arm long  --gpu 0 --epochs 3000 --k1 0.002 --k2 0.2 --activation relu --hidden-units 256 256
```

### Evaluation of the initial policies

Assume that there exists a folder named `result`.

Eight-points reaching in the learned targets:
```
python src/rl/evaluate.py --model results/result_h_short/pyt_save/model.pt --norender --out result/result_h_short.pickle
python src/rl/evaluate.py --model results/result_s_short/pyt_save/model.pt --norender --out result/result_s_short.pickle
python src/rl/evaluate.py --model results/result_h_long/pyt_save/model.pt  --norender --out result/result_h_long.pickle
python src/rl/evaluate.py --model results/result_s_long/pyt_save/model.pt  --norender --out result/result_s_long.pickle
```

### Synergy visualization

Motor synergy waveforms:
```
python src/synergetic/show_synergy.py --filename result/result_h_short.pickle --n-synergies 4
mv result/motor_synergies.pdf result/motor_synergies-horizontal_short.pdf

python src/synergetic/show_synergy.py --filename result/result_s_short.pickle --n-synergies 4
mv result/motor_synergies.pdf result/motor_synergies-sagittal_short.pdf

python src/synergetic/show_synergy.py --filename result/result_h_long.pickle --n-synergies 4
mv result/motor_synergies.pdf result/motor_synergies-horizontal_long.pdf

python src/synergetic/show_synergy.py --filename result/result_s_long.pickle --n-synergies 4
mv result/motor_synergies.pdf result/motor_synergies-sagittal_long.pdf
```

### Optimization by varying numbers of motor synergies

```
./src/synergetic/optimization_all2.sh
```

### Visualization of the results

Performance maps:
```
python src/synergetic/show_performance_map2.py --targets frontal --arm short
mv result/performance_map.pdf result/performance_map-frontal-short.pdf

python src/synergetic/show_performance_map2.py --targets frontal --arm long
mv result/performance_map.pdf result/performance_map-frontal-long.pdf
```
