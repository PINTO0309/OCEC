#!/usr/bin/env python

from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import copy
import json
import cv2
import math
import time
from pprint import pprint
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser, ArgumentTypeError
from typing import Tuple, Optional, List, Dict, Any
import importlib.util
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

AVERAGE_HEAD_WIDTH: float = 0.16 + 0.10 # 16cm + Margin Compensation

BOX_COLORS = [
    [(216, 67, 21),"Front"],
    [(255, 87, 34),"Right-Front"],
    [(123, 31, 162),"Right-Side"],
    [(255, 193, 7),"Right-Back"],
    [(76, 175, 80),"Back"],
    [(33, 150, 243),"Left-Back"],
    [(156, 39, 176),"Left-Side"],
    [(0, 188, 212),"Left-Front"],
]

# The pairs of classes you want to join
# (there is some overlap because there are left and right classes)
EDGES = [
    (21, 22), (21, 22),  # collarbone -> shoulder (left and right)
    (21, 23),            # collarbone -> solar_plexus
    (22, 24), (22, 24),  # shoulder -> elbow (left and right)
    (22, 30), (22, 30),  # shoulder -> hip_joint (left and right)
    (24, 25), (24, 25),  # elbow -> wrist (left and right)
    (23, 29),            # solar_plexus -> abdomen
    (29, 30), (29, 30),  # abdomen -> hip_joint (left and right)
    (30, 31), (30, 31),  # hip_joint -> knee (left and right)
    (31, 32), (31, 32),  # knee -> ankle (left and right)
]

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

CROP_SIZE_RULES = {
    'open': {
        'min_height': 5,
        'max_height': 50,
        'min_width': 10,
        'max_width': 80,
    },
    'closed': {
        'min_height': 5,
        'max_height': 30,
        'min_width': 10,
        'max_width': 80,
    },
}

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    generation: int = -1 # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1 # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    head_pose: int = -1 # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
    is_used: bool = False
    person_id: int = -1
    track_id: int = -1

class SimpleSortTracker:
    """Minimal SORT-style tracker based on IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: List[Dict[str, Any]] = []
        self.frame_index = 0

    @staticmethod
    def _iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        if inter_w == 0 or inter_h == 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def update(self, boxes: List[Box]) -> None:
        self.frame_index += 1

        for box in boxes:
            box.track_id = -1

        if not boxes and not self.tracks:
            return

        iou_matrix = None
        if self.tracks and boxes:
            iou_matrix = np.zeros((len(self.tracks), len(boxes)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                track_bbox = track['bbox']
                for d_idx, box in enumerate(boxes):
                    det_bbox = (box.x1, box.y1, box.x2, box.y2)
                    iou_matrix[t_idx, d_idx] = self._iou(track_bbox, det_bbox)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: List[Tuple[int, int]] = []

        if iou_matrix is not None and iou_matrix.size > 0:
            while True:
                best_track = -1
                best_det = -1
                best_iou = self.iou_threshold
                for t_idx in range(len(self.tracks)):
                    if t_idx in matched_tracks:
                        continue
                    for d_idx in range(len(boxes)):
                        if d_idx in matched_detections:
                            continue
                        iou = float(iou_matrix[t_idx, d_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_track = t_idx
                            best_det = d_idx
                if best_track == -1:
                    break
                matched_tracks.add(best_track)
                matched_detections.add(best_det)
                matches.append((best_track, best_det))

        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det_box = boxes[d_idx]
            track['bbox'] = (det_box.x1, det_box.y1, det_box.x2, det_box.y2)
            track['missed'] = 0
            track['last_seen'] = self.frame_index
            det_box.track_id = track['id']

        surviving_tracks: List[Dict[str, Any]] = []
        for idx, track in enumerate(self.tracks):
            if idx in matched_tracks:
                surviving_tracks.append(track)
                continue
            track['missed'] += 1
            if track['missed'] <= self.max_age:
                surviving_tracks.append(track)
        self.tracks = surviving_tracks

        for d_idx, det_box in enumerate(boxes):
            if d_idx in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            det_box.track_id = track_id
            self.tracks.append(
                {
                    'id': track_id,
                    'bbox': (det_box.x1, det_box.y1, det_box.x2, det_box.y2),
                    'missed': 0,
                    'last_seen': self.frame_index,
                }
            )

        if not boxes:
            return

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._keypoint_th = keypoint_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            print(f'{Color.GREEN("Enabled ONNX ExecutionProviders:")}')
            pprint(f'{self._providers}')

            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3

        elif self._runtime in ['ai_edge_litert', 'tensorflow']:
            if self._runtime == 'ai_edge_litert':
                from ai_edge_litert.interpreter import Interpreter
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['ai_edge_litert', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

class DEIMv2(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'deimv2_dinov3_x_wholebody34_1750query_n_batch.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for DEIMv2. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for DEIMv2

        obj_class_score_th: Optional[float]
            Object score threshold. Default: 0.35

        attr_class_score_th: Optional[float]
            Attributes score threshold. Default: 0.70

        keypoint_th: Optional[float]
            Keypoints score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            keypoint_th=keypoint_th,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in DEIMv2
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in DEIMv2

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, atrributes, is_used=False]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0][0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )
        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]. [instances, [batchno, classid, score, x1, y1, x2, y2]].

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, attributes, is_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        box_score_threshold: float = min([self._obj_class_score_th, self._attr_class_score_th, self._keypoint_th])

        if len(boxes) > 0:
            scores = boxes[:, 5:6]
            keep_idxs = scores[:, 0] > box_score_threshold
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                # Object filter
                for box, score in zip(boxes_keep, scores_keep):
                    classid = int(box[0])
                    x_min = int(max(0, box[1]) * image_width)
                    y_min = int(max(0, box[2]) * image_height)
                    x_max = int(min(box[3], 1.0) * image_width)
                    y_max = int(min(box[4], 1.0) * image_height)
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    result_boxes.append(
                        Box(
                            classid=classid,
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                            generation=-1, # -1: Unknown, 0: Adult, 1: Child
                            gender=-1, # -1: Unknown, 0: Male, 1: Female
                            handedness=-1, # -1: Unknown, 0: Left, 1: Right
                            head_pose=-1, # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
                        )
                    )
                # Object filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [0,5,6,7,16,17,18,19,20,26,27,28,33] and box.score >= self._obj_class_score_th) or box.classid not in [0,5,6,7,16,17,18,19,20,26,27,28,33]
                ]
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [1,2,3,4,8,9,10,11,12,13,14,15] and box.score >= self._attr_class_score_th) or box.classid not in [1,2,3,4,8,9,10,11,12,13,14,15]
                ]
                # Keypoint filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [21,22,23,24,25,29,30,31,32] and box.score >= self._keypoint_th) or box.classid not in [21,22,23,24,25,29,30,31,32]
                ]

                # Adult, Child merge
                # classid: 0 -> Body
                #   classid: 1 -> Adult
                #   classid: 2 -> Child
                # 1. Calculate Adult and Child IoUs for Body detection results
                # 2. Connect either the Adult or the Child with the highest score and the highest IoU with the Body.
                # 3. Exclude Adult and Child from detection results
                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=generation_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]
                # Male, Female merge
                # classid: 0 -> Body
                #   classid: 3 -> Male
                #   classid: 4 -> Female
                # 1. Calculate Male and Female IoUs for Body detection results
                # 2. Connect either the Male or the Female with the highest score and the highest IoU with the Body.
                # 3. Exclude Male and Female from detection results
                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=gender_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]
                # HeadPose merge
                # classid: 7 -> Head
                #   classid:  8 -> Front
                #   classid:  9 -> Right-Front
                #   classid: 10 -> Right-Side
                #   classid: 11 -> Right-Back
                #   classid: 12 -> Back
                #   classid: 13 -> Left-Back
                #   classid: 14 -> Left-Side
                #   classid: 15 -> Left-Front
                # 1. Calculate HeadPose IoUs for Head detection results
                # 2. Connect either the HeadPose with the highest score and the highest IoU with the Head.
                # 3. Exclude HeadPose from detection results
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [box for box in result_boxes if box.classid in [8,9,10,11,12,13,14,15]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [8,9,10,11,12,13,14,15]]
                # Left and right hand merge
                # classid: 23 -> Hand
                #   classid: 24 -> Left-Hand
                #   classid: 25 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 26]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [27, 28]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [27, 28]]

                # Keypoints NMS
                # Suppression of overdetection
                # classid: 21 -> collarbone
                # classid: 22 -> shoulder
                # classid: 23 -> solar_plexus
                # classid: 24 -> elbow
                # classid: 25 -> wrist
                # classid: 29 -> abdomen
                # classid: 30 -> hip_joint
                # classid: 31 -> knee
                # classid: 32 -> ankle
                for target_classid in [21,22,23,24,25,29,30,31,32]:
                    keypoints_boxes = [box for box in result_boxes if box.classid == target_classid]
                    filtered_keypoints_boxes = self._nms(target_objs=keypoints_boxes, iou_threshold=0.20)
                    result_boxes = [box for box in result_boxes if box.classid != target_classid]
                    result_boxes = result_boxes + filtered_keypoints_boxes
        return result_boxes

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                # Process only unused objects with center Euclidean distance less than or equal to 10.0
                if not target_obj.is_used and distance <= 10.0:
                    # Prioritize high-score objects
                    if target_obj.score >= best_score:
                        # IoU Calculation
                        iou: float = \
                            self._calculate_iou(
                                base_obj=base_obj,
                                target_obj=target_obj,
                            )
                        # Adopt object with highest IoU
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 27:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 28:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _nms(
        self,
        *,
        target_objs: List[Box],
        iou_threshold: float,
    ):
        filtered_objs: List[Box] = []

        # 1. Sorted in order of highest score
        #    key=lambda box: box.score to get the score, and reverse=True to sort in descending order
        sorted_objs = sorted(target_objs, key=lambda box: box.score, reverse=True)

        # 2. Scan the box list after sorting
        while sorted_objs:
            # Extract the first (highest score)
            current_box = sorted_objs.pop(0)

            # If you have already used it, skip it
            if current_box.is_used:
                continue

            # Add to filtered_objs and set the use flag
            filtered_objs.append(current_box)
            current_box.is_used = True

            # 3. Mark the boxes where the current_box and IOU are above the threshold as used or exclude them
            remaining_boxes = []
            for box in sorted_objs:
                if not box.is_used:
                    # Calculating IoU
                    iou_value = self._calculate_iou(base_obj=current_box, target_obj=box)

                    # If the IOU threshold is exceeded, it is considered to be the same object and is removed as a duplicate
                    if iou_value >= iou_threshold:
                        # Leave as used (exclude later)
                        box.is_used = True
                    else:
                        # If the IOU threshold is not met, the candidate is still retained
                        remaining_boxes.append(box)

            # Only the remaining_boxes will be handled in the next loop
            sorted_objs = remaining_boxes

        # 4. Return the box that is left over in the end
        return filtered_objs

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

def list_image_files(dir_path: str) -> List[str]:
    path = Path(dir_path)
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(path.rglob(extension))
    return sorted([str(file) for file in image_files])


class CropIndexer:
    """Manage crop file numbering and folder rotation."""

    def __init__(self, root: Path, start_folder: int, folder_size: int = 1000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.start_folder = start_folder
        self.folder_size = folder_size
        self._count = 0

    def save_crop(self, crop: np.ndarray, base_name: str, detection_idx: int) -> Optional[Path]:
        cleaned_base = base_name.replace(' ', '_')
        folder_number = self.start_folder + (self._count // self.folder_size)
        folder_name = f'{folder_number:09d}'
        folder_path = self.root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        file_name = f'{cleaned_base}_{detection_idx + 1}.png'
        file_path = folder_path / file_name
        success = cv2.imwrite(str(file_path), crop)
        if not success:
            return None
        self._count += 1
        return Path('data') / 'cropped' / folder_name / file_name


class EyeAnalysisTracker:
    """Collect statistics and artifacts for eye-only analysis."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.total_images = 0
        self.images_with_eye = 0
        self.over_detected = 0
        self.heights_by_label: Dict[str, List[int]] = defaultdict(list)
        self.widths_by_label: Dict[str, List[int]] = defaultdict(list)
        self.crops_per_label: Counter = Counter()
        self.annotation_entries: List[Tuple[str, int]] = []

    @property
    def images_without_eye(self) -> int:
        return max(0, self.total_images - self.images_with_eye)

    def register_image(self, detection_count: int) -> None:
        self.total_images += 1
        if detection_count > 0:
            self.images_with_eye += 1
        if detection_count >= 3:
            self.over_detected += 1

    def register_crop(
        self,
        label_name: str,
        classid: int,
        relative_path: Path,
        height_px: int,
        width_px: int,
    ) -> None:
        posix_path = relative_path.as_posix()
        self.annotation_entries.append((posix_path, classid))
        if height_px > 0:
            self.heights_by_label[label_name].append(height_px)
        if width_px > 0:
            self.widths_by_label[label_name].append(width_px)
        self.crops_per_label[label_name] += 1

    def write_annotation_csv(self) -> None:
        annotation_path = self.output_root / 'annotation.csv'
        merged_entries: Dict[str, int] = {}
        if annotation_path.exists():
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if ',' not in line:
                        continue
                    rel_path, classid_str = line.split(',', 1)
                    rel_path = rel_path.strip()
                    classid_str = classid_str.strip()
                    try:
                        merged_entries[rel_path] = int(classid_str)
                    except ValueError:
                        continue
        for rel_path, classid in self.annotation_entries:
            merged_entries[rel_path] = classid
        sorted_entries = sorted(merged_entries.items(), key=lambda item: item[0])
        with open(annotation_path, 'w', encoding='utf-8') as f:
            for rel_path, classid in sorted_entries:
                f.write(f'{rel_path},{classid}\n')

    def write_histograms(self) -> None:
        for label_name in sorted(set(self.heights_by_label.keys()) | set(self.widths_by_label.keys())):
            heights = self.heights_by_label.get(label_name, [])
            widths = self.widths_by_label.get(label_name, [])
            if not heights and not widths:
                continue
            output_path = self.output_root / f'{label_name}_eye_size_hist.png'
            save_eye_histogram(label_name=label_name, heights=heights, widths=widths, output_path=output_path)

    def print_summary(self) -> None:
        print(Color.GREEN('Eye-only detection summary'))
        print(f'  Total images: {self.total_images}')
        print(f'  Images with detection: {self.images_with_eye}')
        print(f'  Images without detection: {self.images_without_eye}')
        print(f'  Images with >=3 detections: {self.over_detected}')
        if self.crops_per_label:
            print('  Crops per label:')
            for label_name, count in sorted(self.crops_per_label.items()):
                print(f'    {label_name}: {count}')


def save_eye_histogram(label_name: str, heights: List[int], widths: List[int], output_path: Path) -> None:
    """Create histogram png with mean/median overlays for height and width."""
    def _prepare_bins(values: List[int]) -> List[int]:
        if not values:
            return [0, 1]
        v_min = min(values)
        v_max = max(values)
        if v_min == v_max:
            return [v_min - 0.5, v_min + 0.5]
        return list(range(int(v_min), int(v_max) + 2))

    def _plot(ax, data: List[int], label: str, color: str) -> None:
        bins = _prepare_bins(data)
        counts, _, _ = ax.hist(data, bins=bins, color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel(f'{label} (px)')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
        if not data:
            return
        mean_val = float(np.mean(data))
        median_val = float(np.median(data))
        y_max = max(counts) if len(counts) > 0 else 0
        y_base = y_max * 0.95 if y_max > 0 else 1.0
        median_color = '#dd8452'
        mean_color = '#55a868'
        ax.axvline(median_val, color=median_color, linestyle='--', linewidth=1.5)
        ax.text(
            median_val,
            y_base,
            f'Median: {median_val:.1f}',
            rotation=90,
            va='top',
            ha='right',
            color=median_color,
            fontsize=10,
        )
        ax.axvline(mean_val, color=mean_color, linestyle='-', linewidth=1.5)
        ax.text(
            mean_val,
            y_base * 0.85,
            f'Mean: {mean_val:.1f}',
            rotation=90,
            va='top',
            ha='left',
            color=mean_color,
            fontsize=10,
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f'{label_name.capitalize()} eye crop size distribution')
    _plot(axes[0], heights, 'Height', '#4c72b0')
    _plot(axes[1], widths, 'Width', '#c44e52')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)


def load_label_from_json(image_path: Path) -> Optional[Tuple[str, int]]:
    json_path = image_path.with_suffix('.json')
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    label_value = meta.get('label')
    if label_value == 'closed_eyes':
        return 'closed', 0
    if label_value == 'open_eyes':
        return 'open', 1
    return None


def resolve_video_label(video_path: Path) -> Optional[Tuple[str, int]]:
    stem = video_path.stem.lower()
    if 'open' in stem:
        return 'open', 0
    if 'closed' in stem:
        return 'closed', 1
    return None


def crop_eye_region(image: np.ndarray, box: Box) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1 = max(0, min(int(box.x1), w))
    y1 = max(0, min(int(box.y1), h))
    x2 = max(0, min(int(box.x2), w))
    y2 = max(0, min(int(box.y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def is_valid_crop(label_name: str, height_px: int, width_px: int) -> bool:
    rules = CROP_SIZE_RULES.get(label_name)
    if rules is None:
        return True
    if height_px < rules['min_height'] or height_px > rules['max_height']:
        return False
    if width_px < rules['min_width'] or width_px > rules['max_width']:
        return False
    return True


def run_eye_analysis(
    *,
    model: DEIMv2,
    image_dirs: List[Path],
    video_paths: List[Path],
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
) -> None:
    output_root = Path('data') / 'cropped'
    tracker = EyeAnalysisTracker(output_root=output_root)
    dataset_crop_indexer = CropIndexer(root=tracker.output_root, start_folder=1)
    video_crop_indexer = CropIndexer(root=tracker.output_root, start_folder=100000001)

    processed_any = False
    for images_dir in image_dirs:
        if not images_dir.exists() or not images_dir.is_dir():
            print(Color.YELLOW(f'Skipping missing image directory: {images_dir}'))
            continue
        print(Color.GREEN(f'Processing image directory: {images_dir}'))
        process_images_dir_for_eye_analysis(
            model=model,
            images_dir=images_dir,
            tracker=tracker,
            crop_indexer=dataset_crop_indexer,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        processed_any = True

    for video_path in video_paths:
        if not video_path.exists():
            print(Color.YELLOW(f'Skipping missing video file: {video_path}'))
            continue
        print(Color.GREEN(f'Processing video file: {video_path}'))
        process_video_for_eye_analysis(
            model=model,
            video_path=video_path,
            tracker=tracker,
            crop_indexer=video_crop_indexer,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        processed_any = True

    if not processed_any:
        print(Color.RED('ERROR: No valid sources were processed for eye analysis.'))
        return

    tracker.write_histograms()
    tracker.write_annotation_csv()
    tracker.print_summary()


def process_images_dir_for_eye_analysis(
    *,
    model: DEIMv2,
    images_dir: Path,
    tracker: EyeAnalysisTracker,
    crop_indexer: CropIndexer,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
) -> None:
    image_paths = list_image_files(str(images_dir))
    if not image_paths:
        print(Color.YELLOW(f'No image files found under {images_dir}'))
        return
    total_images = len(image_paths)
    for idx, image_path_str in enumerate(image_paths):
        image_path = Path(image_path_str)
        label_info = load_label_from_json(image_path)
        if label_info is None:
            print(Color.YELLOW(f'Skipping {image_path} because label metadata is unavailable.'))
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(Color.YELLOW(f'Failed to read image: {image_path}'))
            continue
        label_name, classid = label_info
        boxes = model(
            image=image,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        eye_boxes = [box for box in boxes if box.classid == 17]
        eye_boxes.sort(key=lambda box: box.score, reverse=True)
        detection_count = len(eye_boxes)
        tracker.register_image(detection_count)
        if detection_count == 0:
            continue
        for det_idx, box in enumerate(eye_boxes[:2]):
            crop = crop_eye_region(image, box)
            if crop is None or crop.size == 0:
                continue
            crop_height, crop_width = crop.shape[0], crop.shape[1]
            if not is_valid_crop(label_name, crop_height, crop_width):
                continue
            saved_rel_path = crop_indexer.save_crop(crop, base_name=image_path.stem, detection_idx=det_idx)
            if saved_rel_path is None:
                continue
            tracker.register_crop(
                label_name=label_name,
                classid=classid,
                relative_path=saved_rel_path,
                height_px=crop_height,
                width_px=crop_width,
            )
        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            print(f'Processed {idx + 1}/{total_images} images.')


def process_video_for_eye_analysis(
    *,
    model: DEIMv2,
    video_path: Path,
    tracker: EyeAnalysisTracker,
    crop_indexer: CropIndexer,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
) -> None:
    label_info = resolve_video_label(video_path)
    if label_info is None:
        print(Color.YELLOW(f'Could not determine label/classid mapping for video: {video_path}'))
        return
    label_name, classid = label_info
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(Color.RED(f'ERROR: Failed to open video {video_path}'))
        return
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            boxes = model(
                image=frame,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
            eye_boxes = [box for box in boxes if box.classid == 17]
            eye_boxes.sort(key=lambda box: box.score, reverse=True)
            detection_count = len(eye_boxes)
            tracker.register_image(detection_count)
            if detection_count == 0:
                continue
            base_name = f'{video_path.stem}_{frame_idx:08d}'
            for det_idx, box in enumerate(eye_boxes[:2]):
                crop = crop_eye_region(frame, box)
                if crop is None or crop.size == 0:
                    continue
                crop_height, crop_width = crop.shape[0], crop.shape[1]
                if not is_valid_crop(label_name, crop_height, crop_width):
                    continue
                saved_rel_path = crop_indexer.save_crop(crop, base_name=base_name, detection_idx=det_idx)
                if saved_rel_path is None:
                    continue
                tracker.register_crop(
                    label_name=label_name,
                    classid=classid,
                    relative_path=saved_rel_path,
                    height_px=crop_height,
                    width_px=crop_width,
                )
            if frame_idx % 100 == 0:
                print(f'Processed {frame_idx} frames from {video_path}.')
    finally:
        cap.release()

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)

def distance_euclid(p1: Tuple[int,int], p2: Tuple[int,int]) -> float:
    """2点 (x1, y1), (x2, y2) のユークリッド距離を返す"""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def draw_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color=(0,255,255),
    max_dist_threshold=500.0
):
    """
    与えられた boxes (各クラスIDの関節候補) を基に、EDGESで定義された親子を
    「もっとも近い距離のペアから順番に」接合していく。ただし、
    classid=0 (人物) のバウンディングボックス内にあるキーポイント同士のみを
    接続対象とする。
    """
    # -------------------------
    # 1) 人物ボックスに ID を付与する
    # -------------------------
    person_boxes = [b for b in boxes if b.classid == 0]
    for i, pbox in enumerate(person_boxes):
        # 便宜上、Boxクラスに person_id 属性がないので動的に付与する例
        pbox.person_id = i

    # -------------------------------------------------
    # 2) キーポイントがどの人物ボックスに属するか判断して person_id を記録
    #    （複数人のバウンディングボックスが重なっている場合は、
    #      先に見つかったものを採用、など適宜ルールを決める）
    # -------------------------------------------------
    keypoint_ids = {21,22,23,24,25,29,30,31,32}
    for box in boxes:
        if box.classid in keypoint_ids:
            box.person_id = -1
            for pbox in person_boxes:
                if (pbox.x1 <= box.cx <= pbox.x2) and (pbox.y1 <= box.cy <= pbox.y2):
                    box.person_id = pbox.person_id
                    break

    # -------------------------
    # 3) クラスIDごとに仕分け
    # -------------------------
    classid_to_boxes: Dict[int, List[Box]] = {}
    for b in boxes:
        classid_to_boxes.setdefault(b.classid, []).append(b)

    edge_counts = Counter(EDGES)

    # 結果のラインを入れる
    lines_to_draw = []

    # ユークリッド距離計算の簡易関数
    def distance_euclid(p1, p2):
        import math
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    # 各 (pid, cid) ペアに対してグルーピング
    for (pid, cid), repeat_count in edge_counts.items():
        parent_list = classid_to_boxes.get(pid, [])
        child_list  = classid_to_boxes.get(cid, [])

        if not parent_list or not child_list:
            continue

        # 親クラスIDが21 or 29の時はEDGESに書かれている回数(=repeat_count)だけマッチ可
        # それ以外は1回だけ
        for_parent = repeat_count if (pid in [21, 29]) else 1

        parent_capacity = [for_parent]*len(parent_list)  # 親ごとに繋げる上限

        # 子は常に1回のみ
        child_used = [False]*len(child_list)

        # 距離が小さいペアから順に確定していくために、全ペアの距離を計算
        pair_candidates = []
        for i, pbox in enumerate(parent_list):
            for j, cbox in enumerate(child_list):
                # ここで "同じ person_id 同士であること" をチェック
                if (pbox.person_id is not None) and (cbox.person_id is not None) and (pbox.person_id == cbox.person_id):

                    dist = distance_euclid((pbox.cx, pbox.cy), (cbox.cx, cbox.cy))
                    if dist <= max_dist_threshold:
                        pair_candidates.append((dist, i, j))

        # 距離の小さい順に並べ替え
        pair_candidates.sort(key=lambda x: x[0])

        # 貪欲に割り当て
        for dist, i, j in pair_candidates:
            if parent_capacity[i] > 0 and (not child_used[j]):
                # 親iがまだマッチ可能 & 子jが未使用ならマッチ確定
                pbox = parent_list[i]
                cbox = child_list[j]

                lines_to_draw.append(((pbox.cx, pbox.cy), (cbox.cx, cbox.cy)))
                parent_capacity[i] -= 1
                child_used[j] = True

    # -------------------------
    # 4) ラインを描画
    # -------------------------
    for (pt1, pt2) in lines_to_draw:
        cv2.line(image, pt1, pt2, color, thickness=2)

def main():
    parser = ArgumentParser()

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 2:
            raise ArgumentTypeError(f"Invalid Value: {ivalue}. Please specify an integer of 2 or greater.")
        return ivalue

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='deimv2_dinov3_x_wholebody34_1750query_n_batch.onnx',
        help='ONNX/TFLite file path for DEIMv2.',
    )
    group_v_or_i = parser.add_mutually_exclusive_group(required=False)
    group_v_or_i.add_argument(
        '-v',
        '--video',
        type=str,
        help='Video file path or camera index.',
    )
    group_v_or_i.add_argument(
        '-i',
        '--images_dir',
        type=str,
        help='jpg, png images folder path.',
    )
    parser.add_argument(
        '-ep',
        '-oep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cpu',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-it',
        '--inference_type',
        type=str,
        choices=['fp16', 'int8'],
        default='fp16',
        help='Inference type. Default: fp16',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    parser.add_argument(
        '-dwk',
        '--disable_waitKey',
        action='store_true',
        help=\
            'Disable cv2.waitKey(). '+
            'When you want to process a batch of still images, '+
            ' disable key-input wait and process them continuously.',
    )
    parser.add_argument(
        '-ost',
        '--object_socre_threshold',
        type=float,
        default=0.35,
        help=\
            'The detection score threshold for object detection. Default: 0.35',
    )
    parser.add_argument(
        '-ast',
        '--attribute_socre_threshold',
        type=float,
        default=0.70,
        help=\
            'The attribute score threshold for object detection. Default: 0.70',
    )
    parser.add_argument(
        '-kst',
        '--keypoint_threshold',
        type=float,
        default=0.30,
        help=\
            'The keypoint score threshold for object detection. Default: 0.30',
    )
    parser.add_argument(
        '-kdm',
        '--keypoint_drawing_mode',
        type=str,
        choices=['dot', 'box', 'both'],
        default='dot',
        help='Key Point Drawing Mode. Default: dot',
    )
    parser.add_argument(
        '-ebm',
        '--enable_bone_drawing_mode',
        action='store_true',
        help=\
            'Enable bone drawing mode. (Press B on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dnm',
        '--disable_generation_identification_mode',
        action='store_true',
        help=\
            'Disable generation identification mode. (Press N on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dgm',
        '--disable_gender_identification_mode',
        action='store_true',
        help=\
            'Disable gender identification mode. (Press G on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dlr',
        '--disable_left_and_right_hand_identification_mode',
        action='store_true',
        help=\
            'Disable left and right hand identification mode. (Press H on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhm',
        '--disable_headpose_identification_mode',
        action='store_true',
        help=\
            'Disable HeadPose identification mode. (Press P on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-ea',
        '--eye_analysis',
        action='store_true',
        help='Enable eye-only analysis workflow (cropping, histograms, annotation export).',
    )
    parser.add_argument(
        '-drc',
        '--disable_render_classids',
        type=int,
        nargs="*",
        default=[],
        help=\
            'Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19',
    )
    parser.add_argument(
        '-efm',
        '--enable_face_mosaic',
        action='store_true',
        help=\
            'Enable face mosaic.',
    )
    parser.add_argument(
        '-dtk',
        '--disable_tracking',
        action='store_true',
        help=\
            'Disable instance tracking. (Press R on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dti',
        '--disable_trackid_overlay',
        action='store_true',
        help=\
            'Disable TrackID overlay. (Press T on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhd',
        '--disable_head_distance_measurement',
        action='store_true',
        help=\
            'Disable Head distance measurement. (Press M on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-oyt',
        '--output_yolo_format_text',
        action='store_true',
        help=\
            'Output YOLO format texts and images.',
    )
    parser.add_argument(
        '-bblw',
        '--bounding_box_line_width',
        type=check_positive,
        default=2,
        help=\
            'Bounding box line width. Default: 2',
    )
    parser.add_argument(
        '-chf',
        '--camera_horizontal_fov',
        type=int,
        default=90,
        help=\
            'Camera horizontal FOV. Default: 90',
    )
    args = parser.parse_args()

    # runtime check
    model_file: str = args.model
    model_dir_path = os.path.dirname(os.path.abspath(model_file))
    model_ext: str = os.path.splitext(model_file)[1][1:].lower()
    runtime: str = None
    if model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif model_ext == 'tflite':
        if is_package_installed('ai_edge_litert'):
            runtime = 'ai_edge_litert'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: ai_edge_litert or tensorflow is not installed.'))
            sys.exit(0)
    video: str = args.video
    images_dir: str = args.images_dir
    disable_waitKey: bool = args.disable_waitKey
    object_socre_threshold: float = args.object_socre_threshold
    attribute_socre_threshold: float = args.attribute_socre_threshold
    keypoint_threshold: float = args.keypoint_threshold
    keypoint_drawing_mode: str = args.keypoint_drawing_mode
    enable_bone_drawing_mode: bool = args.enable_bone_drawing_mode
    disable_generation_identification_mode: bool = args.disable_generation_identification_mode
    disable_gender_identification_mode: bool = args.disable_gender_identification_mode
    disable_left_and_right_hand_identification_mode: bool = args.disable_left_and_right_hand_identification_mode
    disable_headpose_identification_mode: bool = args.disable_headpose_identification_mode
    eye_analysis_enabled: bool = args.eye_analysis
    if not eye_analysis_enabled and video is None and images_dir is None:
        parser.error('Either --video or --images_dir must be specified unless --eye_analysis is enabled.')
    disable_render_classids: List[int] = args.disable_render_classids
    enable_face_mosaic: bool = args.enable_face_mosaic
    enable_tracking: bool = not args.disable_tracking
    enable_trackid_overlay: bool = not args.disable_trackid_overlay
    enable_head_distance_measurement: bool = not args.disable_head_distance_measurement
    output_yolo_format_text: bool = args.output_yolo_format_text
    execution_provider: str = args.execution_provider
    inference_type: str = args.inference_type
    inference_type = inference_type.lower()
    bounding_box_line_width: int = args.bounding_box_line_width
    camera_horizontal_fov: int = args.camera_horizontal_fov
    providers: List[Tuple[str, Dict] | str] = None

    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        ep_type_params = {}
        if inference_type == 'fp16':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        elif inference_type == 'int8':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                    "trt_int8_enable": True,
                    "trt_int8_calibration_table_name": "calibration.flatbuffers",
                }
        else:
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': f'{model_dir_path}',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                } | ep_type_params,
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ]

    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    # Model initialization
    model = DEIMv2(
        runtime=runtime,
        model_path=model_file,
        obj_class_score_th=object_socre_threshold,
        attr_class_score_th=attribute_socre_threshold,
        keypoint_th=keypoint_threshold,
        providers=providers,
    )

    if eye_analysis_enabled:
        image_dirs_for_analysis: List[Path] = []
        video_paths_for_analysis: List[Path] = []

        if images_dir is not None:
            image_dirs_for_analysis.append(Path(images_dir))

        if video is not None:
            if is_parsable_to_int(video):
                print(Color.YELLOW('Skipping camera index for eye analysis; only file paths are supported.'))
            else:
                video_paths_for_analysis.append(Path(video))

        if not image_dirs_for_analysis and not video_paths_for_analysis:
            default_dir = Path('data') / 'extracted' / 'train'
            image_dirs_for_analysis.append(default_dir)
            default_videos = [
                Path('real_data') / 'open.mp4',
                Path('real_data') / 'closed.mp4',
            ]
            video_paths_for_analysis.extend(default_videos)

        # Deduplicate while preserving order
        seen_dirs = set()
        unique_image_dirs: List[Path] = []
        for candidate in image_dirs_for_analysis:
            resolved = candidate
            if resolved in seen_dirs:
                continue
            unique_image_dirs.append(resolved)
            seen_dirs.add(resolved)

        seen_videos = set()
        unique_video_paths: List[Path] = []
        for candidate in video_paths_for_analysis:
            resolved = candidate
            if resolved in seen_videos:
                continue
            unique_video_paths.append(resolved)
            seen_videos.add(resolved)

        run_eye_analysis(
            model=model,
            image_dirs=unique_image_dirs,
            video_paths=unique_video_paths,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        return

    file_paths: List[str] = None
    cap = None
    video_writer = None
    if images_dir is not None:
        file_paths = list_image_files(dir_path=images_dir)
    else:
        cap = cv2.VideoCapture(
            int(video) if is_parsable_to_int(video) else video
        )
        disable_video_writer: bool = args.disable_video_writer
        if not disable_video_writer:
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                filename='output.mp4',
                fourcc=fourcc,
                fps=cap_fps,
                frameSize=(w, h),
            )

    file_paths_count = -1
    movie_frame_count = 0
    white_line_width = bounding_box_line_width
    colored_line_width = white_line_width - 1
    tracker = SimpleSortTracker()
    track_color_cache: Dict[int, np.ndarray] = {}
    tracking_enabled_prev = enable_tracking
    while True:
        image: np.ndarray = None
        if file_paths is not None:
            file_paths_count += 1
            if file_paths_count <= len(file_paths) - 1:
                image = cv2.imread(file_paths[file_paths_count])
            else:
                break
        else:
            res, image = cap.read()
            if not res:
                break
            movie_frame_count += 1

        debug_image = copy.deepcopy(image)
        debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()
        boxes = model(
            image=debug_image,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        elapsed_time = time.perf_counter() - start_time

        if file_paths is None:
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        body_boxes = [box for box in boxes if box.classid == 0]
        current_tracking_enabled = enable_tracking
        if current_tracking_enabled:
            if not tracking_enabled_prev:
                tracker = SimpleSortTracker()
                track_color_cache.clear()
            tracker.update(body_boxes)
            active_track_ids = {track['id'] for track in tracker.tracks}
            stale_ids = [tid for tid in track_color_cache.keys() if tid not in active_track_ids]
            for tid in stale_ids:
                track_color_cache.pop(tid, None)
        else:
            if tracking_enabled_prev:
                tracker = SimpleSortTracker()
                track_color_cache.clear()
            for box in boxes:
                box.track_id = -1
        tracking_enabled_prev = current_tracking_enabled

        # Draw bounding boxes
        for box in boxes:
            classid: int = box.classid
            color = (255,255,255)

            if classid in disable_render_classids:
                continue

            if classid == 0:
                # Body
                if not disable_gender_identification_mode:
                    # Body
                    if box.gender == 0:
                        # Male
                        color = (255,0,0)
                    elif box.gender == 1:
                        # Female
                        color = (139,116,225)
                    else:
                        # Unknown
                        color = (0,200,255)
                else:
                    # Body
                    color = (0,200,255)
            elif classid == 5:
                # Body-With-Wheelchair
                color = (0,200,255)
            elif classid == 6:
                # Body-With-Crutches
                color = (83,36,179)
            elif classid == 7:
                # Head
                if not disable_headpose_identification_mode:
                    color = BOX_COLORS[box.head_pose][0] if box.head_pose != -1 else (216,67,21)
                else:
                    color = (0,0,255)
            elif classid == 16:
                # Face
                color = (0,200,255)
            elif classid == 17:
                # Eye
                color = (255,0,0)
            elif classid == 18:
                # Nose
                color = (0,255,0)
            elif classid == 19:
                # Mouth
                color = (0,0,255)
            elif classid == 20:
                # Ear
                color = (203,192,255)

            elif classid == 21:
                # Collarbone
                color = (0,0,255)
            elif classid == 22:
                # Shoulder
                color = (255,0,0)
            elif classid == 23:
                # Solar_plexus
                color = (252,189,107)
            elif classid == 24:
                # Elbow
                color = (0,255,0)
            elif classid == 25:
                # Wrist
                color = (0,0,255)

            elif classid == 26:
                if not disable_left_and_right_hand_identification_mode:
                    # Hands
                    if box.handedness == 0:
                        # Left-Hand
                        color = (0,128,0)
                    elif box.handedness == 1:
                        # Right-Hand
                        color = (255,0,255)
                    else:
                        # Unknown
                        color = (0,255,0)
                else:
                    # Hands
                    color = (0,255,0)

            elif classid == 29:
                # abdomen
                color = (0,0,255)
            elif classid == 30:
                # hip_joint
                color = (255,0,0)
            elif classid == 31:
                # Knee
                color = (0,0,255)
            elif classid == 32:
                # ankle
                color = (255,0,0)

            elif classid == 33:
                # Foot
                color = (250,0,136)

            if (classid == 0 and not disable_gender_identification_mode) \
                or (classid == 7 and not disable_headpose_identification_mode) \
                or (classid == 26 and not disable_left_and_right_hand_identification_mode) \
                or classid == 16 \
                or classid in [21,22,23,24,25,29,30,31,32]:

                # Body
                if classid == 0:
                    if box.gender == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Head
                elif classid == 7:
                    if box.head_pose == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Face
                elif classid == 16:
                    if enable_face_mosaic:
                        w = int(abs(box.x2 - box.x1))
                        h = int(abs(box.y2 - box.y1))
                        small_box = cv2.resize(debug_image[box.y1:box.y2, box.x1:box.x2, :], (3,3))
                        normal_box = cv2.resize(small_box, (w,h))
                        if normal_box.shape[0] != abs(box.y2 - box.y1) \
                            or normal_box.shape[1] != abs(box.x2 - box.x1):
                                normal_box = cv2.resize(small_box, (abs(box.x2 - box.x1), abs(box.y2 - box.y1)))
                        debug_image[box.y1:box.y2, box.x1:box.x2, :] = normal_box
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Hands
                elif classid == 26:
                    if box.handedness == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Shoulder, Elbow, Knee
                elif classid in [21,22,23,24,25,29,30,31,32]:
                    if keypoint_drawing_mode in ['dot', 'both']:
                        cv2.circle(debug_image, (box.cx, box.cy), 4, (255,255,255), -1)
                        cv2.circle(debug_image, (box.cx, box.cy), 3, color, -1)
                    if keypoint_drawing_mode in ['box', 'both']:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)

            else:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

            # TrackID text
            if enable_trackid_overlay and classid == 0 and box.track_id > 0:
                track_text = f'ID: {box.track_id}'
                text_x = max(box.x1 - 5, 0)
                text_y = box.y1 - 30
                if text_y < 20:
                    text_y = min(box.y2 + 25, debug_image_h - 10)
                cached_color = track_color_cache.get(box.track_id)
                if isinstance(cached_color, np.ndarray):
                    text_color = tuple(int(np.clip(v, 0, 255)) for v in cached_color.tolist())
                else:
                    text_color = color if isinstance(color, tuple) else (0, 200, 255)
                cv2.putText(
                    debug_image,
                    track_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (10, 10, 10),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    track_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            # Attributes text
            generation_txt = ''
            if box.generation == -1:
                generation_txt = ''
            elif box.generation == 0:
                generation_txt = 'Adult'
            elif box.generation == 1:
                generation_txt = 'Child'

            gender_txt = ''
            if box.gender == -1:
                gender_txt = ''
            elif box.gender == 0:
                gender_txt = 'M'
            elif box.gender == 1:
                gender_txt = 'F'

            attr_txt = f'{generation_txt}({gender_txt})' if gender_txt != '' else f'{generation_txt}'

            headpose_txt = BOX_COLORS[box.head_pose][1] if box.head_pose != -1 else ''
            attr_txt = f'{attr_txt} {headpose_txt}' if headpose_txt != '' else f'{attr_txt}'

            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            handedness_txt = ''
            if box.handedness == -1:
                handedness_txt = ''
            elif box.handedness == 0:
                handedness_txt = 'L'
            elif box.handedness == 1:
                handedness_txt = 'R'
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            # Head distance
            if enable_head_distance_measurement and classid == 7:
                focalLength: float = 0.0
                if (camera_horizontal_fov > 90):
                    # Fisheye Camera (Equidistant Model)
                    focalLength = debug_image_w / (camera_horizontal_fov * (math.pi / 180))
                else:
                    # Normal camera (Pinhole Model)
                    focalLength = debug_image_w / (2 * math.tan((camera_horizontal_fov / 2) * (math.pi / 180)))
                # Meters
                distance = (AVERAGE_HEAD_WIDTH * focalLength) / abs(box.x2 - box.x1)

                cv2.putText(
                    debug_image,
                    f'{distance:.3f} m',
                    (
                        box.x1+5 if box.x1 < debug_image_w else debug_image_w-50,
                        box.y1+20 if box.y1-5 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{distance:.3f} m',
                    (
                        box.x1+5 if box.x1 < debug_image_w else debug_image_w-50,
                        box.y1+20 if box.y1-15 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (10, 10, 10),
                    1,
                    cv2.LINE_AA,
                )

            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (255, 255, 255),
            #     2,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     color,
            #     1,
            #     cv2.LINE_AA,
            # )

        # Draw skeleton
        if enable_bone_drawing_mode:
            draw_skeleton(image=debug_image, boxes=boxes, color=(0, 255, 255), max_dist_threshold=300)

        if file_paths is not None:
            basename = os.path.basename(file_paths[file_paths_count])
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{basename}', debug_image)

        if file_paths is not None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_i.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_o.png', debug_image)
            with open(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        elif file_paths is None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{movie_frame_count:08d}.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_i.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_o.png', debug_image)
            with open(f'output/{movie_frame_count:08d}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

        if video_writer is not None:
            video_writer.write(debug_image)
            # video_writer.write(image)

        cv2.imshow("test", debug_image)

        key = cv2.waitKey(1) & 0xFF if file_paths is None or disable_waitKey else cv2.waitKey(0) & 0xFF
        if key == ord('\x1b'): # 27, ESC
            break
        elif key == ord('b'): # 98, B, Bone drawing mode switch
            enable_bone_drawing_mode = not enable_bone_drawing_mode
        elif key == ord('n'): # 110, N, Generation mode switch
            disable_generation_identification_mode = not disable_generation_identification_mode
        elif key == ord('g'): # 103, G, Gender mode switch
            disable_gender_identification_mode = not disable_gender_identification_mode
        elif key == ord('p'): # 112, P, HeadPose mode switch
            disable_headpose_identification_mode = not disable_headpose_identification_mode
        elif key == ord('h'): # 104, H, HandsLR mode switch
            disable_left_and_right_hand_identification_mode = not disable_left_and_right_hand_identification_mode
        elif key == ord('k'): # 107, K, Keypoints mode switch
            if keypoint_drawing_mode == 'dot':
                keypoint_drawing_mode = 'box'
            elif keypoint_drawing_mode == 'box':
                keypoint_drawing_mode = 'both'
            elif keypoint_drawing_mode == 'both':
                keypoint_drawing_mode = 'dot'
        elif key == ord('r'): # 114, R, Tracking mode switch
            enable_tracking = not enable_tracking
            if enable_tracking and not enable_trackid_overlay:
                enable_trackid_overlay = True
        elif key == ord('t'): # 116, T, TrackID overlay mode switch
            enable_trackid_overlay = not enable_trackid_overlay
            if not enable_tracking:
                enable_trackid_overlay = False
        elif key == ord('m'): # 109, M, Head distance measurement mode switch
            enable_head_distance_measurement = not enable_head_distance_measurement

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

    try:
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
