from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import Registry


__all__ = ['DATASETS']

##################################################################
#                        mmvpr.datasets                          #
##################################################################

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
    locations=['mmvpr.datasets'],
)

##################################################################
#                         mmvpr.models                           #
##################################################################

# Neural network modules inheriting `nn.Module`.
MODELS = Registry(
    'model',
    parent=MMENGINE_MODELS,
    locations=['mmvpr.models'],
)
# Batch augmentations like `Mixup` and `CutMix`.
BATCH_AUGMENTS = Registry(
    'batch augment',
    locations=['mmvpr.models'],
)

##################################################################
#                       mmvpr.evaluation                         #
##################################################################

# Metrics to evaluate the model prediction results.
METRICS = Registry(
    'metric',
    parent=MMENGINE_METRICS,
    locations=['mmvpr.evaluation'],
)