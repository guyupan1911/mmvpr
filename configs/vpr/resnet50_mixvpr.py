_base_ = [
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(type='BatchResize', scale=(320,320)),
    # dict(type='RandomResizedCrop', scale=224),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackVPRInputs'),
]

train_dataloader = dict(
    batch_size=100,
    num_workers=8,
    dataset=dict(
        type='GSVCities',
        dataset_path='/home/yuxuanhuang/projects/OpenVPRLab/data/train/gsv-cities-light',
        img_per_place = 4,
        random_sample_from_each_place=True,
        pipeline=train_pipeline,
        serialize_data=False),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320,320)),
    dict(type='PackInputs'),
]

test_dataloader = dict(
    batch_size=256,
    num_workers=16,
    dataset=dict(
        type='PittsburghDataset',
        dataset_path='/home/yuxuanhuang/projects/OpenVPRLab/data/val/pitts30k-val/',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_dataloader = dict(
    batch_size=256,
    num_workers=16,
    dataset=dict(
        type='MapillarySLSDataset',
        dataset_path='/home/yuxuanhuang/projects/OpenVPRLab/data/val/msls-val/',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False)
)


model = dict(
    type='VisualPlaceRecognizer',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        frozen_stages=2,
        out_indices=(2, ),
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(
        type='VPRHead',
        aggregator=dict(type='MixVPR'),
        loss=dict(type='VPRLoss'))
)


# optimizer
optim_wrapper=dict(
    type='AmpOptimWrapper',
    # 如果你想要使用 BF16，请取消下面一行的代码注释
    dtype='float16',  # 可用值： ('float16', 'bfloat16', None)
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.001))

# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,  # Updated by iterations
         begin=0,
         end=1500),  # Warm up for the first 1500 iterations
    # dict(
    #     type='MultiStepLR', by_epoch=True, milestones=[10, 20, 30], gamma=0.1)
] 

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

test_evaluator = dict(type='VprMetric', topk=(1, 5, 10))
val_evaluator = test_evaluator

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
