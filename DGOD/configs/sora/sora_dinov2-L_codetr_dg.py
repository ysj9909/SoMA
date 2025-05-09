_base_ = [
    '../_base_/models/dinov2_codetr.py',
    '../_base_/datasets/dg_diverse_weather_1024x1024.py',
    '../_base_/default_runtime.py',
]

rank = 16
model = dict(type="PEFTBackboneCoDETR",
             backbone=dict(
                 lora_cfg=dict(
                     type="sora",
                     r = rank,
                     first_eigen = False,
                     lora_alpha = rank,
                     lora_dropout = 0.,
                     lora_weight_init = "sora",
                     target_modules = ['q', 'k', 'v', 'proj', 'fc1', 'fc2'],
                     start_lora_idx = 8,
                    #  merge_weights=True,
                 ),
             ),
             query_head=dict(
                 num_query=900,
             ),
        )

# use 4 gpus
train_dataloader = dict(batch_size=2)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0002, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.5, decay_mult=0.2),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "norm": dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-2, by_epoch=False, begin=0, end=1500),
    dict(
        type='MultiStepLR',
        begin=1500,
        end=40000,
        by_epoch=False,
        milestones=[24000, 32000],
        gamma=0.1),
    dict(
        type='CosineAnnealingParamScheduler',
        param_name='weight_decay',
        eta_min=0.00001,
        by_epoch=False,
        begin=0,
        end=40000)
]

# training schedule for 40k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=8
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
    # visualization=dict(type="DetVisualizationHook", draw=True, interval=1, show=True),
)
log_processor = dict(by_epoch=False)
custom_hooks = [
    dict(type='WeightDecayLoggingHook'),
]
find_unused_parameters=True