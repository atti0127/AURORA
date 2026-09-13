import os
import io
import copy
import itertools
import json
import logging
import contextlib
from collections import OrderedDict

import numpy as np
import torch

import pycocotools.mask as mask_util
from multiprocessing import freeze_support
from fvcore.common.file_io import PathManager
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from .datasets.avis_api.avos import AVOS

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import aviseval


_PRIMARY_METRICS = (
    ("FA", "FSLA (FA)"),
    ("HOTA", "HOTA"),
    ("AP_all", "mAP (AP_all)"),
)


def format_avis_metrics(metrics, dataset_name=None, gt_file=None):
    """Format AVIS metrics without changing the official computation."""
    lines = ["AVIS Evaluation Metrics", "======================="]
    if dataset_name:
        lines.append("Dataset: {}".format(dataset_name))
    if gt_file:
        lines.append("Ground truth: {}".format(gt_file))
    lines.extend(["", "Primary metrics", "---------------"])
    for key, label in _PRIMARY_METRICS:
        lines.append("{}: {:.2f}".format(label, float(metrics[key])))

    remaining = sorted(set(metrics) - {key for key, _ in _PRIMARY_METRICS})
    if remaining:
        lines.extend(["", "Additional metrics", "------------------"])
        for key in remaining:
            value = metrics[key]
            if isinstance(value, (int, np.integer)):
                rendered = str(int(value))
            elif isinstance(value, (float, np.floating)):
                rendered = "{:.2f}".format(float(value))
            else:
                rendered = str(value)
            lines.append("{}: {}".format(key, rendered))
    lines.extend(["", "All scores are percentages; higher is better.", ""])
    return "\n".join(lines)


def write_avis_metrics(metrics, result_dir, dataset_name=None, gt_file=None):
    """Write a readable companion to the official result JSON."""
    metrics_path = os.path.join(result_dir, "metrics.txt")
    with PathManager.open(metrics_path, "w") as output:
        output.write(format_avis_metrics(metrics, dataset_name, gt_file))
        output.flush()
    return metrics_path


def _configure_gt_file(dataset_config, gt_file):
    gt_folder, gt_name = os.path.split(os.fspath(gt_file))
    if gt_folder:
        dataset_config["GT_FOLDER"] = os.path.abspath(gt_folder)
    dataset_config["GT_File"] = gt_name or os.fspath(gt_file)


def eval_track(out_dir, gt_file):
    freeze_support()

    # Command line interface:
    default_eval_config = aviseval.Evaluator.get_default_eval_config()
    default_dataset_config = aviseval.datasets.AVIS.get_default_dataset_config()
    default_dataset_config['TRACKERS_FOLDER'] = out_dir
    _configure_gt_file(default_dataset_config, gt_file)
    default_metrics_config = {'METRICS': ['TrackMAP', 'HOTA']}   # 'CLEAR', 'Identity'
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = aviseval.Evaluator(eval_config)
    dataset_list = [aviseval.datasets.AVIS(dataset_config)]
    metrics_list = []
    for metric in [aviseval.metrics.TrackMAP, aviseval.metrics.HOTA]:
        if metric.get_name() in metrics_config['METRICS']:
            if metric == aviseval.metrics.TrackMAP:
                default_track_map_config = metric.get_default_metric_config()
                default_track_map_config['USE_TIME_RANGES'] = False
                default_track_map_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                                           [128 ** 2, 256 ** 2],
                                                           [256 ** 2, 1e5 ** 2]]
                metrics_list.append(metric(default_track_map_config))
            else:
                metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    return output_res


def instances_to_coco_json_video(inputs, outputs):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]
    video_length = inputs[0]["length"]

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]
    score_features = outputs.get("pred_score_features")

    avis_results = []
    for instance_id, (s, l, m) in enumerate(zip(scores, labels, masks)):
        segms = [
            mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
            for _mask in m
        ]
        for rle in segms:
            rle["counts"] = rle["counts"].decode("utf-8")

        res = {
            "video_id": video_id,
            "score": s,
            "category_id": l,
            "segmentations": segms,
        }
        if score_features is not None:
            for name, values in score_features.items():
                res[name] = values[instance_id]
        avis_results.append(res)

    return avis_results



class AVISEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            tasks=None,
            distributed=True,
            output_dir=None,
            *,
            use_fast_impl=True,
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        self._cpu_device = torch.device("cpu")

        self.dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._gt_json_file = json_file
        with contextlib.redirect_stdout(io.StringIO()):
            self._avis_api = AVOS(json_file)

        self._do_evaluation = "annotations" in self._avis_api.dataset


    def reset(self):
        self._predictions = []


    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        prediction = instances_to_coco_json_video(inputs, outputs)
        self._predictions.extend(prediction)


    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            if not comm.is_main_process():
                return {}
            predictions = list(itertools.chain.from_iterable(predictions))
        else:
            predictions = self._predictions

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for AVIS format ...")

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id

            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in predictions:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        o_d = None
        if self._output_dir:
            o_d = os.path.join(self._output_dir, "results")
            os.makedirs(os.path.join(o_d, "model_final"), exist_ok=True)
            file_path = os.path.join(o_d, "model_final", "results.json")

            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        assert o_d != None
        output_res = eval_track(o_d, self._gt_json_file)
        metrics = output_res["AVIS"]["model_final"]
        self._results["segm"] = metrics
        metrics_path = write_avis_metrics(
            metrics,
            os.path.join(o_d, "model_final"),
            dataset_name=self.dataset_name,
            gt_file=self._gt_json_file,
        )
        self._logger.info("Saving readable metrics to %s", metrics_path)
