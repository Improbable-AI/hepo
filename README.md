Official Going Beyond Heurisitics by Imposing Policy Improvement As A Constraint
=
**Paper link**: https://openreview.net/pdf?id=vBGMbFgvsX

**Abstract:** In many reinforcement learning (RL) applications, incorporating heuristic rewards alongside the task reward is crucial for achieving desirable performance. Heuristics encode prior human knowledge about how a task should be done, providing valuable hints for RL algorithms. However, such hints may not be optimal, limiting the performance of learned policies. The currently established way of using heuristics is to modify the heuristic reward in a manner that ensures that the optimal policy learned with it remains the same as the optimal policy for the task reward (i.e., optimal policy invariance). However, these methods often fail in practical scenarios with limited training data. We found that while optimal policy invariance ensures convergence to the best policy based on task rewards, it doesn't guarantee better performance than policies trained with biased heuristics under a finite data regime, which is impractical. In this paper, we introduce a new principle tailored for finite data settings. Instead of enforcing optimal policy invariance, we train a policy that combines task and heuristic rewards and ensures it outperforms the heuristic-trained policy. As such, we prevent policies from merely exploiting heuristic rewards without improving the task reward. Our experiments on robotic locomotion, helicopter control, and manipulation tasks demonstrate that our method consistently outperforms the heuristic policy, regardless of the heuristic rewards' quality. 


## Reproduce experiments in `IsaacGym`

**1. Clone this repoisitory: `git clone https://github.com/Improbable-AI/hepo`**

**2. Install the dependicies:**
```bash
cd hepo
conda env create -f environment.yaml
pip install -e .
```

**3. Install `IsaacGym`**
   1. Download and install Isaac Gym Preview 4 from [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
   2. Unzip the file:
   ```bash
   tar -xf IsaacGym_Preview_4_Package.tar.gz
   ```
   3. Install the Python package:
   ```bash
   cd isaacgym/python && pip install -e .
   ```
   4. Verify the installation by running an example:

   ```bash
   python examples/1080_balls_of_solitude.py
   ```
   5. For troubleshooting, check the documentation at `isaacgym/docs/index.html`.


## Running the benchmarks in robotic tasks
The following are the commands to run the experiments for the methods `H-only`, `J-only`, and `HEPO` (see our paper for details). In short, `H-only` is PPO trained with heursitic reward provided with the benchmarking task and `J-only` is PPO trained with the task reward only. `HEPO` is trained with both heuristic and task rewards.

#### J-only
```bash
python train.py ext_scheme='success' task=$task max_iterations=$max_iterations seed=$seed 
```

#### H-only
```bash
python train.py ext_scheme='total' task=$task max_iterations=$max_iterations seed=$seed 
```

#### HEPO
```bash
python train.py lmbd=1. alpha=0. update_alpha_gae=True use_hepo=True ext_scheme='success' int_scheme='total' alpha_lr=0.0001 task=$task max_iterations=$max_iterations seed=$seed
```

For `$task` and `$max_iterations`, please follow the table below and subsititute the values to the commands above. The available `IsaacGym` and `Bi-Dex` tasks and the suggested `max_iterations` are the following:
| Task            | max_iterations |
|-----------------|----------------|
| Ant             | 3000           |
| Anymal          | 1500           |
| Quadcopter      | 3000           |
| Ingenuity       | 800            |
| Humanoid        | 10000          |
| FrankaCabinet   | 1500           |
| FrankaCubeStack | 5000           |
| AllegroHand     | 15000          |
| ShadowHand      | 15000          |
| ShadowHandSpin               | 10000          |
| ShadowHandUpsideDown         | 10000          |
| ShadowHandBlockStack         | 10000          |
| ShadowHandBottleCap          | 10000          |
| ShadowHandCatchAbreast       | 10000          |
| ShadowHandCatchOver2Underarm | 10000          |
| ShadowHandCatchUnderarm      | 10000          |
| ShadowHandDoorCloseInward    | 10000          |
| ShadowHandDoorCloseOutward   | 10000          |
| ShadowHandDoorOpenInward     | 10000          |
| ShadowHandDoorOpenOutward    | 10000          |
| ShadowHandGraspAndPlace      | 10000          |
| ShadowHandKettle             | 10000          |
| ShadowHandLiftUnderarm       | 10000          |
| ShadowHandPen                | 10000          |
| ShadowHandOver               | 10000          |
| ShadowHandPushBlock          | 10000          |
| ShadowHandReOrientation      | 10000          |
| ShadowHandScissors           | 10000          |
| ShadowHandSwingCup           | 10000          |
| ShadowHandSwitch             | 10000          |
| ShadowHandTwoCatchUnderarm   | 10000          |

For logging, we use Weight & Bias. Please add the following arguments to the command if you'd like to log the experimental results:
```bash
wandb_activate=True
wandb_project=${W&B project name}
experiment=${W&B run name}
```
The average task reward of `H-only` and `J-only` is stored at `rewards/ppo_metric`, and that of `HEPO` is stored at `reawrds/hepo_metric`.

One example command with logging should look like:
```bash
python train.py ext_scheme='success' task=Ant max_iterations=3000 seed=0 wandb_activate=True wandb_project=Ant experiment=J_only
```

# Citing this paper
```
@inproceedings{
lee2024going,
title={Going Beyond Heuristics by Imposing Policy Improvement as a Constraint},
author={Chi-Chang Lee* and Zhang-Wei Hong* and Pulkit Agrawal},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=vBGMbFgvsX}
}
```