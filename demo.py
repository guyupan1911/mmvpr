from mmvpr.registry import DATASETS

dataset_type = 'ImageNet'
cfg = dict(
        type=dataset_type,
        data_root='data/imagenet',
        split='val')

imagenet_dataset = DATASETS.build(cfg)

print(f'imagenet_dataset: {imagenet_dataset}')

