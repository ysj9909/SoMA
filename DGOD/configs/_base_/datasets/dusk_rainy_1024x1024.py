dr_type = 'VOCDataset'
dr_root = 'data/diverseWeather/dusk_rainy'

dr_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(720, 1280), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
dr_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dr = dict(
    type=dr_type,
    data_root=dr_root,
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=dr_train_pipeline,
    backend_args=None
)
test_dr = dict(
    type=dr_type,
    data_root=dr_root,
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    test_mode=True,
    pipeline=dr_test_pipeline,
    backend_args=None
)