# dataset config
_base_ = [
    "../_base_/datasets/dg_citys2bm_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/models/dinov2_mask2former.py",
]

crop_size = (1024, 1024)
rank = 16
model = dict(type="PEFTBackboneEncoderDecoder",
    backbone=dict(
        img_size=1024,
        init_cfg=dict(
            checkpoint="checkpoints/dinov2_sora_converted_1024x1024.pth",
        ),
        lora_cfg=dict(
            type="sora",
            r = rank,
            first_eigen = False,
            lora_alpha = rank,
            lora_dropout = 0.,
            lora_weight_init = "sora",
            target_modules = ['q', 'k', 'v', 'proj', 'fc1', 'fc2'],
            start_lora_idx = 8,
        ),
    ),
    data_preprocessor=dict(
        size=crop_size,
    ),
    test_cfg=dict(
        crop_size=(1024, 1024),
        stride=(683, 683),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=4096,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.5, decay_mult=1.0),
            "query_embed": embed_multi,
            "query_feat": embed_multi,
            "level_embed": embed_multi,
            "norm": dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-2, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", 
         eta_min=0, 
         power=0.9, 
         begin=1500, 
         end=40000, 
         by_epoch=False),
    dict(
        type='CosineAnnealingParamScheduler',
        param_name='weight_decay',
        eta_min=0.00001,
        by_epoch=False,
        begin=0,
        end=40000)
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=6
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
custom_hooks = [
    dict(type='WeightDecayLoggingHook'),
    dict(type='EMAHook', begin_iter=16000, priority='NORMAL')
]
find_unused_parameters = True