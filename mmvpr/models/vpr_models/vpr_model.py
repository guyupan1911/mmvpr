from typing import List, Optional

import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmvpr.registry import MODELS
from mmvpr.structures import DataSample


@MODELS.register_module()
class VisualPlaceRecognizer(BaseModel):
    """ Visual Place Recognition Model

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmvpr.models.backbones`.
        aggregator (dict, optional): The aggregation layer to process features from
            backbone. See :mod:`mmvpr.models.aggregators`. Defaults to None.
        loss_function: 
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                backbone: dict,
                head: dict,
                pretrained: Optional[str] = None,
                data_preprocessor: Optional[dict] = None,
                init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        
        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'VprDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        
        super(VisualPlaceRecognizer, self).__init__(
            init_cfg = init_cfg, data_preprocessor = data_preprocessor
        )

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.head = head

        # If the model needs to load pretrain weights from a third party,
        # the key can be modified with this hook
        if hasattr(self.backbone, '_checkpoint_filter'):
            self._register_load_state_dict_pre_hook(
                self.backbone._checkpoint_filter)        
    
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
    
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode: "{mode}".')

    def extract_feat(self, inputs):
        x = self.backbone(inputs)
        return x

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)
    
    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples)