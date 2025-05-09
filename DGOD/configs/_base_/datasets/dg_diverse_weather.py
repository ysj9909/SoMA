_base_ = [
    "./daytime_clear.py",
    "./night_sunny.py",
    "./dusk_rainy.py",
    "./night_rainy.py",
    "./daytime_foggy.py",
]

img_scales = [(800, 1333), (720, 1280), (600, 1067), (400, 666)]
tta_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [dict(type='LoadAnnotations', with_bbox=True)], [
            dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]

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