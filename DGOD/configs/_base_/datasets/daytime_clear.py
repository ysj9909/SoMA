dc_type = 'VOCDataset'
dc_root = 'data/diverseWeather/daytime_clear'
backend_args = None
image_size = (720, 1280)
# dc_train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(720, 1280), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
dc_train_pipeline= [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type='PackDetInputs')
]
dc_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(720, 1280), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dc = dict(
    type=dc_type,
    data_root=dc_root,
    ann_file='VOC2007/ImageSets/Main/train.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=dc_train_pipeline,
    backend_args=backend_args
)
test_dc = dict(
    type=dc_type,
    data_root=dc_root,
    ann_file='VOC2007/ImageSets/Main/test.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    test_mode=True,
    pipeline=dc_test_pipeline,
    backend_args=backend_args
)