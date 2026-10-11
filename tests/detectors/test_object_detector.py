from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from gardendome.detectors.object_detector import ObjectDetector

LABELS_PATH = Path(__file__).parents[2] / "models/ssd_mobilenet_v1/labelmap.txt"
PERSON = 0
CUP = 46


def make_detector(boxes, class_ids, scores, score_threshold=0.5):
    interpreter = MagicMock()
    interpreter.get_input_details.return_value = [
        {"index": 0, "shape": np.array([1, 300, 300, 3], dtype=np.int32)}
    ]
    interpreter.get_output_details.return_value = [{"index": 1}, {"index": 2}, {"index": 3}]
    outputs = {
        1: np.array([boxes], dtype=np.float32),
        2: np.array([class_ids], dtype=np.float32),
        3: np.array([scores], dtype=np.float32),
    }
    interpreter.get_tensor.side_effect = lambda index: outputs[index]

    with patch(
        "gardendome.detectors.object_detector.Interpreter", return_value=interpreter
    ):
        detector = ObjectDetector(
            model_path="detect.tflite",
            num_threads=4,
            labels_path=LABELS_PATH,
            score_threshold=score_threshold,
        )
    return detector, interpreter


def colour_frame(height, width):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_detect_returns_label_score_and_pixel_bbox():
    detector, _ = make_detector(
        boxes=[[0.25, 0.1, 0.75, 0.4]], class_ids=[CUP], scores=[0.9]
    )
    [detection] = detector.detect(colour_frame(480, 640))
    assert detection.label == "cup"
    assert detection.score == np.float32(0.9)
    assert detection.bbox == (64, 120, 192, 240)


def test_detect_stops_at_first_score_below_threshold():
    detector, _ = make_detector(
        boxes=[[0, 0, 1, 1]] * 3,
        class_ids=[PERSON, CUP, PERSON],
        scores=[0.9, 0.6, 0.3],
    )
    detections = detector.detect(colour_frame(480, 640))
    assert [d.label for d in detections] == ["person", "cup"]


def test_detect_returns_empty_when_no_score_meets_threshold():
    detector, _ = make_detector(boxes=[[0, 0, 1, 1]], class_ids=[PERSON], scores=[0.4])
    assert detector.detect(colour_frame(480, 640)) == []


def test_detect_sends_model_sized_rgb_batch():
    detector, interpreter = make_detector(
        boxes=[[0, 0, 1, 1]], class_ids=[PERSON], scores=[0.4]
    )
    frame = colour_frame(480, 640)
    frame[:, :] = (255, 0, 0)  # pure blue in BGR

    detector.detect(frame)

    index, input_data = interpreter.set_tensor.call_args.args
    assert index == 0
    assert input_data.shape == (1, 300, 300, 3)
    assert tuple(input_data[0, 0, 0]) == (0, 0, 255)  # pure blue in RGB
