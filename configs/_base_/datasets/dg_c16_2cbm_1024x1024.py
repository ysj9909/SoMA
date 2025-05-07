9083
_base_ = [
    "./bdd100k_1024x1024.py",
    # "./cityscapes_1024x1024.py",
    "./mapillary_1024x1024.py",
]
cityscapes_type = "CityscapesDataset"
cityscapes_root = "data/cityscapes/"
cityscapes_crop_size = (1024, 1024)
cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type="Resize", scale=(2048, 1024)),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type="RandomCrop", crop_size=cityscapes_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
cityscapes_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    ann_file='train_1_16_v1.txt',
    data_prefix=dict(
        img_path="leftImg8bit/train",
        seg_map_path="gtFine/train",
    ),
    pipeline=cityscapes_train_pipeline,
)
val_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/val",
        seg_map_path="gtFine/val",
    ),
    pipeline=cityscapes_test_pipeline,
)
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_cityscapes,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            val_cityscapes,
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
