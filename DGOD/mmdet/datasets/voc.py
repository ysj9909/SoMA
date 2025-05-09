# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes': ('bus', 'bike', 'car', 'motor', 'person', 'rider', 'truck'),
        'palette': [(0, 255, 0), (255, 165, 0), (0, 0, 255), (197, 226, 255),
                    (255, 0, 0), (255, 255, 0), (128, 0, 128)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDADataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes': ('bus', 'bike', 'car', 'motor', 'person', 'rider', 'truck'),
        # ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        #  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        #  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 255, 0), (255, 165, 0), (0, 0, 255), (197, 226, 255),
                    (255, 0, 0), (255, 255, 0), (128, 0, 128)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None
