ns_type = 'VOCDataset'
ns_root = 'data/diverseWeather/Night-Sunny'

ns_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(720, 1280), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
ns_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(720, 1280), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_ns = dict(
    type=ns_type,
    data_root=ns_root,
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=ns_train_pipeline,
    backend_args=None
)
test_ns = dict(
    type=ns_type,
    data_root=ns_root,
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    test_mode=True,
    pipeline=ns_test_pipeline,
    backend_args=None
)