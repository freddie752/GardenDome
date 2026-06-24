import numpy as np
from unittest.mock import MagicMock
from camera.feed import Picamera2Feed


def make_camera(frame):
    camera = MagicMock()
    camera.capture_array.return_value = frame
    return camera


def test_get_frame_returns_grayscale_array():
    camera = make_camera(np.zeros((100, 100, 3), dtype=np.uint8))
    frame = Picamera2Feed(camera).get_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 2


def test_get_frame_correct_rgb_to_gray_conversion():
    red_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    red_frame[:, :, 0] = 255  # R channel
    frame = Picamera2Feed(make_camera(red_frame)).get_frame()
    assert frame.mean() > 50
