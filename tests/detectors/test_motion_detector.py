import numpy as np
import pytest
from gardendome.detectors.motion_detector import MotionDetector


def greyscale_frame(height, width, value):
    return np.full((height, width), value, dtype=np.uint8)


def test_raises_on_non_greyscale_previous_frame():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    grey_frame = greyscale_frame(100, 100, 0)
    with pytest.raises(ValueError):
        detector.has_motion(bgr_frame, grey_frame)


def test_raises_on_non_greyscale_current_frame():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    grey_frame = greyscale_frame(100, 100, 0)
    bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.has_motion(grey_frame, bgr_frame)


def test_no_motion_on_identical_frames():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    frame = greyscale_frame(100, 100, 128)
    assert detector.has_motion(frame, frame.copy()) is False


def test_no_motion_when_diff_below_threshold():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 100)
    current = greyscale_frame(100, 100, 105)
    assert detector.has_motion(previous, current) is False


def test_no_motion_when_diff_equals_threshold():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 100)
    current = greyscale_frame(100, 100, 110)
    assert detector.has_motion(previous, current) is False


def test_motion_detected_when_diff_above_threshold():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 100)
    current = greyscale_frame(100, 100, 150)
    assert detector.has_motion(previous, current) is True


def test_no_motion_when_fraction_not_exceeded():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 0)
    current = greyscale_frame(100, 100, 0)
    current[:5, :] = 255
    assert detector.has_motion(previous, current) is False


def test_motion_detected_when_fraction_exceeded():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 0)
    current = greyscale_frame(100, 100, 0)
    current[:20, :] = 255
    assert detector.has_motion(previous, current) is True


def test_locate_motion_returns_none_when_no_motion():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    frame = greyscale_frame(100, 100, 128)
    assert detector.locate_motion(frame, frame.copy()) is None


def test_locate_motion_returns_none_when_fraction_not_exceeded():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 0)
    current = greyscale_frame(100, 100, 0)
    current[:5, :] = 255
    assert detector.locate_motion(previous, current) is None


def test_locate_motion_bounding_box_covers_motion_region():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.05)
    previous = greyscale_frame(100, 100, 0)
    current = greyscale_frame(100, 100, 0)
    current[20:40, 30:70] = 255
    x, y, w, h = detector.locate_motion(previous, current)
    assert x == 30
    assert y == 20
    assert w == 40
    assert h == 20
