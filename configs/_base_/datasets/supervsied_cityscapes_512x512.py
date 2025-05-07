cityscapes_type = "CityscapesDataset"
cityscapes_root = "data/cityscapes/"
cityscapes_crop_size = (512, 512)
cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=cityscapes_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
cityscapes_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/train",
        seg_map_path="gtFine/train",
    ),
    pipeline=cityscapes_train_pipeline,
)
train_dataloader=dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_cityscapes,
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
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_cityscapes,
)
val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU"],
)
test_dataloader=val_dataloader
test_evaluator=val_evaluator
