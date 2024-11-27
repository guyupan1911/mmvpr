from mmvpr.registry import DATASETS
from mmvpr.models import BACKBONES

from mmengine.analysis import get_model_complexity_info

# dataset_type = 'ImageNet'
# cfg = dict(
#         type=dataset_type,
#         data_root='data/imagenet',
#         split='val')

# imagenet_dataset = DATASETS.build(cfg)

# print(f'imagenet_dataset: {imagenet_dataset}')

backbone_cfg = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch')

resnet = BACKBONES.build(backbone_cfg)
input_shape = (3, 480, 480)

analysis_results = get_model_complexity_info(resnet, input_shape)
print(analysis_results['out_table'])

