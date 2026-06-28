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
        detector.detect(bgr_frame, grey_frame)
 
 
def test_raises_on_non_greyscale_current_frame():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    grey_frame = greyscale_frame(100, 100, 0)
    bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.detect(grey_frame, bgr_frame)

 
def test_no_motion_on_identical_frames():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    frame = greyscale_frame(100, 100, 128)
    assert detector.detect(frame, frame.copy()) is False
 
 
def test_no_motion_when_diff_below_threshold():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 100)
    current = greyscale_frame(100, 100, 105)
    assert detector.detect(previous, current) is False
 
 
def test_no_motion_when_diff_equals_threshold():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 100)
    current = greyscale_frame(100, 100, 110)
    assert detector.detect(previous, current) is False
 
 
def test_motion_detected_when_diff_above_threshold():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 100)
    current = greyscale_frame(100, 100, 150)
    assert detector.detect(previous, current) is True
 
 
def test_no_motion_when_fraction_not_exceeded():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 0)
    current = greyscale_frame(100, 100, 0)
    current[:5, :] = 255
    assert detector.detect(previous, current) is False
 
 
def test_motion_detected_when_fraction_exceeded():
    detector = MotionDetector(motion_threshold=10, motion_fraction=0.1)
    previous = greyscale_frame(100, 100, 0)
    current = greyscale_frame(100, 100, 0)
    current[:20, :] = 255
    assert detector.detect(previous, current) is True