_base_ = [
    "./fog-acdc_512x512.py",
    "./night-acdc_512x512.py",
    "./rain-acdc_512x512.py",
    "./snow-acdc_512x512.py",
    "./cityscapes_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_cityscapes}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_night_acdc}},
            {{_base_.val_snow_acdc}},
            {{_base_.val_fog_acdc}},
            {{_base_.val_rain_acdc}},
            {{_base_.val_cityscapes}},
        ],
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.test_night_acdc}},
            {{_base_.test_snow_acdc}},
            {{_base_.test_fog_acdc}},
            {{_base_.test_rain_acdc}},
        ],
    ),
)
val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["night/", "citys/", "fog/", "snow/", "rain/"],
    mean_used_keys=["night/", "fog/", "snow/", "rain/"],
)
test_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU"],
    format_only=True,
    output_dir="/home/wsz/Rein/acdc_results/fft_noEMA_G1234/format_results",
    # output_dir="work_dirs/format_results",
)
# test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
