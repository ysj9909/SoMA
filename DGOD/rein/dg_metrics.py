import os.path as osp
import copy
from typing import List, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger, print_log

from mmdet.registry import METRICS
from mmdet.evaluation.metrics.voc_metric import VOCMetric
from collections import defaultdict


@METRICS.register_module()
class DGVOCMetric(VOCMetric):
    def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
        super().__init__(**kwargs)
        self.dataset_keys = dataset_keys
        if mean_used_keys:
            self.mean_used_keys = mean_used_keys
        else:
            self.mean_used_keys = dataset_keys

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            # TODO: Need to refactor to support LoadAnnotations
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            ann = dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy())

            pred = data_sample['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            dataset_key = "unknown"
            for key in self.dataset_keys:
                if key in data_samples[0]["img_path"]:
                    dataset_key = key
                    break

            self.results.append([dataset_key, ann, dets])

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            dataset_results[result[0]].append(result[1:])
        metrics_type2mean = defaultdict(list)
        for key, key_result in dataset_results.items():
            logger: MMLogger = MMLogger.get_current_instance()
            print_log(f"----------metrics for {key}------------", logger)
            key_metrics = super().compute_metrics(key_result)
            print_log(f"number of samples for {key}: {len(key_result)}")
            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v
                if key in self.mean_used_keys:
                    metrics_type2mean[k].append(v)
        for k, v in metrics_type2mean.items():
            metrics[f"mean_{k}"] = sum(v) / len(v)
        return metrics