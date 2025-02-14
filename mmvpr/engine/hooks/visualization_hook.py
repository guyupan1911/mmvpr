# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
from typing import Optional, Sequence, Dict

from mmengine.fileio import join_path
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer

from mmvpr.registry import HOOKS
from mmvpr.structures import DataSample

import random
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Classification Visualization Hook. Used to visualize validation and
    testing prediction results.

    - If ``out_dir`` is specified, all storage backends are ignored
      and save the image to the ``out_dir``.
    - If ``show`` is True, plot the result image in a window, please
      confirm you are able to access the graphical interface.

    Args:
        enable (bool): Whether to enable this hook. Defaults to False.
        interval (int): The interval of samples to visualize. Defaults to 5000.
        show (bool): Whether to display the drawn image. Defaults to False.
        out_dir (str, optional): directory where painted images will be saved
            in the testing process. If None, handle with the backends of the
            visualizer. Defaults to None.
        **kwargs: other keyword arguments of
            :meth:`mmpretrain.visualization.UniversalVisualizer.visualize_cls`.
    """

    def __init__(self,
                 enable=False,
                 interval: int = 5000,
                 show: bool = False,
                 out_dir: Optional[str] = None,
                 **kwargs):
        self._visualizer: Visualizer = Visualizer.get_current_instance()

        self.enable = enable
        self.interval = interval
        self.show = show
        self.out_dir = out_dir

        self.draw_args = {**kwargs, 'show': show}

    def _draw_samples(self,
                      batch_idx: int,
                      data_batch: dict,
                      data_samples: Sequence[DataSample],
                      step: int = 0) -> None:
        """Visualize every ``self.interval`` samples from a data batch.

        Args:
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DataSample`]): Outputs from model.
            step (int): Global step value to record. Defaults to 0.
        """
        if self.enable is False:
            return

        batch_size = len(data_samples)
        images = data_batch['inputs']
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        # The first index divisible by the interval, after the start index
        first_sample_id = math.ceil(start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, end_idx, self.interval):
            image = images[sample_id - start_idx]
            image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')

            data_sample = data_samples[sample_id - start_idx]
            if 'img_path' in data_sample:
                # osp.basename works on different platforms even file clients.
                sample_name = osp.basename(data_sample.get('img_path'))
            else:
                sample_name = str(sample_id)

            draw_args = self.draw_args
            if self.out_dir is not None:
                draw_args['out_file'] = join_path(self.out_dir,
                                                  f'{sample_name}_{step}.png')

            self._visualizer.visualize_cls(
                image=image,
                data_sample=data_sample,
                step=step,
                name=sample_name,
                **self.draw_args,
            )

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DataSample]) -> None:
        """Visualize every ``self.interval`` samples during validation.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DataSample`]): Outputs from model.
        """
        if isinstance(runner.train_loop, EpochBasedTrainLoop):
            step = runner.epoch
        else:
            step = runner.iter

        self._draw_samples(batch_idx, data_batch, outputs, step=step)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DataSample]) -> None:
        """Visualize every ``self.interval`` samples during test.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): Outputs from model.
        """
        self._draw_samples(batch_idx, data_batch, outputs, step=0)

def visualization(query_indices, preds, dis, gts, img_paths, num_references):
    rows = len(query_indices)
    cols = len(preds[0]) + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.suptitle(f'Query Images and top {len(preds[0])} Predicted References', fontsize=16)

    for i, query_index in enumerate(query_indices):
        query_image = Image.open(img_paths[query_index + num_references])
        axes[i, 0].imshow(query_image)
        axes[i, 0].axis('off')
        # Add text below each image
        axes[i, 0].text(0.5, -0.1, f'{query_index}', fontsize=12, color='black', ha='center', va='top', transform=axes[i, 0].transAxes)  
        for j, pred in enumerate(preds[i]):
            reference_image = Image.open(img_paths[pred])
            axes[i, j+1].imshow(reference_image)
            axes[i, j+1].axis('off')
            similarity_score = 1 / (1 + dis[i, j])
            if pred in gts[i]:
                axes[i, j+1].text(0.5, -0.1, f'True: {similarity_score:.3f}', fontsize=12, color='green', ha='center', va='top', transform=axes[i, j+1].transAxes)  
            else:
                axes[i, j+1].text(0.5, -0.1, f'False: {similarity_score:.3f}', fontsize=12, color='red', ha='center', va='top', transform=axes[i, j+1].transAxes)  
    
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Increase the space between rows and columns
    plt.show()

@HOOKS.register_module()
class VPRVisualizationHook(Hook):
    
    def __init__(self,
                 num_images = 5,
                 topk: int = 5):
        
        self.topk = topk
        self.num_images = num_images

    def after_test_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:

        vis_data = runner.test_evaluator.metrics[0].vis_data
        num_references = vis_data['num_references']
        num_queries = vis_data['num_queries']
        predictions = vis_data['predictions']
        distances = vis_data['distances']
        assert(num_queries == len(predictions))
        ground_truth = vis_data['ground_truth']
        img_paths = vis_data['img_paths']

        random.seed(time.time())
        query_indices = random.sample(range(0, num_queries), k = self.num_images)

        preds = [predictions[idx][:self.topk] for idx in query_indices]
        dis = distances[query_indices, :][:, :self.topk]
        gts = [ground_truth[idx] for idx in query_indices]

        visualization(query_indices, preds, dis, gts, img_paths, num_references)