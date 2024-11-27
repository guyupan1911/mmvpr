from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import Registry
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS


__all__ = ['DATASETS']

#######################################################################
#                        mmpretrain.datasets                          #
#######################################################################

# Datasets like `ImageNet` and `CIFAR10`.
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    locations=['mmvpr.datasets'],
)
# Transforms to process the samples from the dataset.
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmpretrain.datasets'],
)