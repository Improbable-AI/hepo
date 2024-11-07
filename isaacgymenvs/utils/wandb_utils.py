from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict

import os

WANDB_MODE = os.environ.get("WANDB_MODE", "online")
if WANDB_MODE == "offline":
    try:
        import wandb_osh
        from wandb_osh.hooks import TriggerWandbSyncHook
        trigger_sync = TriggerWandbSyncHook()
    except:
        print("Wandb is set as offline, but wandb-osh is not installed. Logs won't be synced online.")
        trigger_sync = lambda: None # Dummy trigger_sync function
else:
    trigger_sync = lambda: None # Dummy trigger_sync function

class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            # wandb.init(
            #     project=cfg.wandb_project,
            #     entity=cfg.wandb_entity,
            #     group=cfg.wandb_group,
            #     tags=cfg.wandb_tags,
            #     sync_tensorboard=True,
            #     id=wandb_unique_id,
            #     name=experiment_name,
            #     resume=True,
            #     settings=wandb.Settings(start_method='fork'),
            # )
            wandb.init(
                project=cfg.wandb_project,
                name=experiment_name,
                settings=wandb.Settings(start_method='fork')
                # sync_tensorboard=True,
            )
            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)
        
        trigger_sync()