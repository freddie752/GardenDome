from dataclasses import dataclass

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple  # x, y, w, h in pixels


class ObjectDetector:
    """Detector for the COCO ssd_mobilenet_v1 TFLite model (uint8 input)."""

    def __init__(self, model_path, num_threads, labels_path, score_threshold):
        self._interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self._interpreter.allocate_tensors()
        self._labels = self._load_labels(labels_path)
        self._score_threshold = score_threshold
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()
        _, self._input_h, self._input_w, _ = self._input_details["shape"]

    def detect(self, bgr_frame):
        """Returns detections above the score threshold, highest score first."""
        frame_h, frame_w = bgr_frame.shape[:2]
        self._interpreter.set_tensor(
            self._input_details["index"], self._prepare_input(bgr_frame)
        )
        self._interpreter.invoke()

        boxes = self._get_output(0)
        class_ids = self._get_output(1)
        scores = self._get_output(2)

        # The model outputs detections in descending score order
        detections = []
        for box, class_id, score in zip(boxes, class_ids, scores):
            if score < self._score_threshold:
                break
            detections.append(
                Detection(
                    label=self._labels[int(class_id)],
                    score=float(score),
                    bbox=self._to_pixel_bbox(box, frame_w, frame_h),
                )
            )
        return detections

    @staticmethod
    def _load_labels(labels_path):
        """Returns the label names indexed by the model's class ids."""
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f]
        # First line is a "???" placeholder; the model's class ids start after it
        return labels[1:]

    def _prepare_input(self, bgr_frame):
        """Returns the frame as an RGB batch of one, resized to the model's input size."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (self._input_w, self._input_h))
        return np.expand_dims(resized, axis=0)

    def _get_output(self, position):
        """Returns the model output at the given position, without the batch dimension."""
        return self._interpreter.get_tensor(self._output_details[position]["index"])[0]

    @staticmethod
    def _to_pixel_bbox(box, frame_w, frame_h):
        """Converts a normalised (ymin, xmin, ymax, xmax) box to pixel (x, y, w, h)."""
        ymin, xmin, ymax, xmax = box
        x, y = int(xmin * frame_w), int(ymin * frame_h)
        w, h = int((xmax - xmin) * frame_w), int((ymax - ymin) * frame_h)
        return x, y, w, h
