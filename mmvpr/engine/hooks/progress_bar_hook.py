# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner

from mmvpr.registry import HOOKS
from mmvpr.structures import DataSample

from tqdm import tqdm

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class ProgressBarHook(Hook):

    def before_train_epoch(self, runner) -> None:
        total_batches = len(runner.train_dataloader)
        self.progress_bar = tqdm(
            total=total_batches,
            desc=f'Train Epoch {runner.epoch}',
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}] {postfix}",
            colour="green",  # Set color of progress bar
            position=0,  # Single progress bar at the top
            leave=True  # Don't clear the progress bar after completion
        )


    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        self.progress_bar.update(1)


    def after_train_epoch(self, runner) -> None:
        if self.progress_bar:
            self.progress_bar.reset()
