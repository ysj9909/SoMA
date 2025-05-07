ur_type = "CityscapesDataset"
ur_root = "data/urbansyn/"
ur_crop_size = (1024, 1024)
ur_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(
    #     type="RandomChoiceResize",
    #     scales=[int(1024 * x * 0.1) for x in range(5, 21)],
    #     resize_type="ResizeShortestEdge",
    #     max_size=4096,
    # ),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    # dict(type="Resize", scale=(2048, 1024)),
    dict(type="RandomCrop", crop_size=ur_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
ur_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2560, 1580), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_ur = dict(
    type=ur_type,
    data_root=ur_root,
    data_prefix=dict(
        img_path="rgb",
        seg_map_path="ss",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=ur_train_pipeline,
)
val_ur = dict(
    type=ur_type,
    data_root=ur_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=ur_test_pipeline,
)