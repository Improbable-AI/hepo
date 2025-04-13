## Installation


#### Install the `isaacgymenvs` package
Then, to run the experiment of HEPO, you can copy our repository and install it by using the pip tool (note that our environment is in ```Python 3.8.16```):
```bash
cd IsaacGymEnvs-dev/
conda env create -f environment.yaml
pip install -e .
```

#### Install Isaac Gym
1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. now install the python package
    ```bash
    cd isaacgym/python && pip install -e .
    ```
4. Verify the installation by try running an example

    ```bash
    python examples/1080_balls_of_solitude.py
    ```
5. For troubleshooting check docs `isaacgym/docs/index.html`

After finishing all the installation processes, go to `IsaacGymEnvs-dev/isaacgymenvs`.


## Running the benchmarks
we conducted our experiments based on the **Isaac Gym** simulator and the **Bi-DexHands** benchmark.
The selected task classes in **Isaac Gym** can be partitioned into 4 groups - Locomotion Tracking (**Anymal**), Locomotion Progressing (**Ant** and **Humanoid**), Helicopter Progressing (**Ingenuity** and **Quadcopter**), and Manipulation Tasks (**FrankaCabinet**, **FrankaCubeStack**, **ShadowHand**, and **AllegroHand**).
In addition, **Bi-DexHands** provides dual dexterous hand manipulation tasks through **Isaac Gym**, reaching human-level sophistication of hand dexterity and bimanual coordination.
Their tasks include **ShadowHandOver**, **ShadowHandCatchUnderarm**, **ShadowHandCatchOver2Underarm**, **ShadowHandCatchAbreast**, **ShadowHandTwoCatchUnderarm**, **ShadowHandLiftUnderarm**, **ShadowHandDoorOpenInward**, **ShadowHandDoorOpenOutward**, **ShadowHandDoorCloseInward**, **ShadowHandDoorCloseOutward**, **ShadowHandSpin**, **ShadowHandUpsideDown**, **ShadowHandBlockStack**, **ShadowHandBottleCap**, **ShadowHandGraspAndPlace**, **ShadowHandKettle**, **ShadowHandPen**, **ShadowHandPushBlock**, **ShadowHandReOrientation**, **ShadowHandScissors**, **ShadowHandSwingCup**.

To train your policies with respect to different methods, you need to specify <code>[TASK]</code>, <code>[TOTAL_EPOCHS]</code>, <code>[SEED]</code>,
and run these lines:

#### J-only
```bash
python train.py task=[TASK] seed=[seed] \
        wandb_activate=True experiment=J-only ext_scheme='success' \
        wandb_project=[TASK] max_iterations=[TOTAL_EPOCHS]
```

#### H-only
```bash
python train.py task=[TASK] seed=[seed] \
        wandb_activate=True experiment=H-only ext_scheme='total' \
        wandb_project=[TASK] max_iterations=[TOTAL_EPOCHS]
```

#### HuRL
```bash
python train.py task=[TASK] seed=[seed] lmbd=1. use_hurl=True \
        wandb_activate=True experiment=HuRL  \
        ext_scheme='success' int_scheme='total' \
        wandb_project=[TASK] max_iterations=[TOTAL_EPOCHS]
```
    
#### PBRS
```bash
python train.py task=[TASK] seed=[seed] lmbd=1. use_pbrs=True \
        wandb_activate=True experiment=PBRS \
        ext_scheme='success' int_scheme='total' \
        wandb_project=[TASK] max_iterations=[TOTAL_EPOCHS]
```

#### HEPO
```bash
python train.py task=[TASK] lmbd=1. alpha=0. \
        use_hepo=True seed=[SEED] wandb_activate=True experiment=HEPO \
        ext_scheme='success' int_scheme='total' alpha_lr=0.001\
        wandb_project=[TASK] max_iterations=[TOTAL_EPOCHS]
```

For example:
```bash
python train.py task=Ant lmbd=1. alpha=0. \
        use_hepo=True seed=1126 wandb_activate=True experiment=HEPO \
        ext_scheme='success' int_scheme='total' alpha_lr=0.001\
        wandb_project=Ant max_iterations=2000
```

The target objective curve will be recorded as <code>rewards/ppo_metric</code> or <code>rewards/hepo_metric</code> on the wandb page.