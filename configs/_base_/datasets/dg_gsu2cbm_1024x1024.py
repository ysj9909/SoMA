_base_ = [
    "./gta_ms_1024x1024.py",
    "./synthia_1024x1024.py",
    "./urbansyn_1024x1024.py",
    "./bdd100k_1024x1024.py",
    "./cityscapes_1024x1024.py",
    "./mapillary_1024x1024.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
          {{_base_.train_gta}},
          {{_base_.train_syn}},
          {{_base_.train_ur}},
        ],
    ),
    
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    # dataset={{_base_.val_bdd}},
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_cityscapes}},
            {{_base_.val_bdd}},
            {{_base_.val_mapillary}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd"]
)
test_evaluator=val_evaluator
