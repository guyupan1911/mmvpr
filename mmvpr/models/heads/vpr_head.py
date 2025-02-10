from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmvpr.registry import MODELS
from mmvpr.structures import DataSample


@MODELS.register_module()
class VPRHead(BaseModule):
    """Visual Place Recognition head.

    Args:
        aggregator (dict): aggregation layer
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """
    def __init__(self,
                 aggregator: dict,
                 loss: dict = dict(type='MultiSimilarityLoss'),
                 init_cfg: Optional[dict] = None):
        super(VPRHead, self).__init__(init_cfg=init_cfg)

        if not isinstance(aggregator, nn.Module):
            aggregator = MODELS.build(aggregator)
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)

        self.aggregation_layer = aggregator
        self.loss_module = loss

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        return self.aggregation_layer(pre_logits)

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        descriptors = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(descriptors, data_samples, **kwargs)
        return losses

    def _get_loss(self, descriptors: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""

        places_ids = torch.cat([i.place_id for i in data_samples])

        # compute loss
        losses = self.loss_module(descriptors, places_ids, **kwargs)

        return losses
    
    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:

        descriptors = self(feats)
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(descriptors, data_samples)
        return predictions

    def _get_predictions(self, descriptors, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        out_data_samples = []
        for data_sample, descriptor in zip(data_samples, descriptors):
            data_sample.set_descriptor(descriptor)
            out_data_samples.append(data_sample)
        return out_data_samples