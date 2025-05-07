syn_type = "CityscapesDataset"
syn_root = "data/synthia/"
syn_crop_size = (1024, 1024)
syn_train_pipeline = [
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
    dict(type="RandomCrop", crop_size=syn_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
syn_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2560, 1580), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_syn = dict(
    type=syn_type,
    data_root=syn_root,
    data_prefix=dict(
        img_path="RGB",
        seg_map_path="GT/LABELS",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=syn_train_pipeline,
)
val_syn = dict(
    type=syn_type,
    data_root=syn_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=syn_test_pipeline,
)