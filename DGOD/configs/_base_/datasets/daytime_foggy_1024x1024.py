df_type = 'VOCDataset'
df_root = 'data/diverseWeather/daytime_foggy'

df_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(720, 1280), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
df_test_pipeline = [
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
train_df = dict(
    type=df_type,
    data_root=df_root,
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=df_train_pipeline,
    backend_args=None
)
test_df = dict(
    type=df_type,
    data_root=df_root,
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    test_mode=True,
    pipeline=df_test_pipeline,
    backend_args=None
)