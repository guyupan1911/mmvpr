from typing import Dict, Optional, Sequence, Union
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmvpr.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class ReloadDatasetHook(Hook):

    def before_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        dataset = runner.train_dataloader.dataset
        dataset.data_list = []
        dataset._fully_initialized = False
        dataset.full_init()
       