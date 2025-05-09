_base_ = [
    "./daytime_clear_1024x1024.py",
    "./night_sunny_1024x1024.py",
    "./dusk_rainy_1024x1024.py",
    "./night_rainy_1024x1024.py",
    "./daytime_foggy_1024x1024.py",
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_dc}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # dataset={{_base_.test_dc}}
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.test_dc}},
            {{_base_.test_ns}},
            {{_base_.test_dr}},
            {{_base_.test_nr}},
            {{_base_.test_df}},
        ]
    )
)
test_dataloader = val_dataloader

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(
    type='DGVOCMetric', metric='mAP', eval_mode='11points', dataset_keys=[
        "daytime_clear",
        "Night-Sunny",
        "dusk_rainy",
        "night_rainy",
        "daytime_foggy"
    ]
)
test_evaluator = val_evaluator