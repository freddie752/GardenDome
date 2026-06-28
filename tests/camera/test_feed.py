import numpy as np
from unittest.mock import MagicMock
from gardendome.camera.feed import Picamera2Feed


def make_camera(frame):
    camera = MagicMock()
    camera.capture_array.return_value = frame
    return camera


def rgb_frame(r=0, g=0, b=0):
    """Helper: solid colour RGB frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = r
    frame[:, :, 1] = g
    frame[:, :, 2] = b
    return frame


def test_get_grey_frame_correct_dimensions():
    feed = Picamera2Feed(make_camera(rgb_frame()))
    frame = feed.get_grey_frame()
    assert frame.shape == (100, 100)


def test_get_grey_frame_rgb_to_grey_conversion():
    feed = Picamera2Feed(make_camera(rgb_frame(r=255)))
    frame = feed.get_grey_frame()
    assert frame.mean() == 76


def test_get_grey_frame_black_input_produces_black_output():
    feed = Picamera2Feed(make_camera(rgb_frame()))
    frame = feed.get_grey_frame()
    assert frame.mean() == 0


def test_get_grey_frame_blur_is_applied():
    sharp_frame = rgb_frame()
    sharp_frame[:, 50:, :] = 255
    feed = Picamera2Feed(make_camera(sharp_frame))
    grey = feed.get_grey_frame()
    edge_pixels = grey[:, 49:52]
    assert edge_pixels.min() > 0
    assert edge_pixels.max() < 255


def test_get_colour_frame_correct_dimensions():
    feed = Picamera2Feed(make_camera(rgb_frame()))
    frame = feed.get_colour_frame()
    assert frame.shape == (100, 100, 3)


def test_get_colour_frame_rgb_to_bgr_conversion():
    feed = Picamera2Feed(make_camera(rgb_frame(r=255)))
    frame = feed.get_colour_frame()
    assert frame[:, :, 0].mean() == 0
    assert frame[:, :, 2].mean() == 255


def test_get_colour_frame_no_blur_applied():
    sharp_frame = rgb_frame()
    sharp_frame[:, 50:, :] = 255
    feed = Picamera2Feed(make_camera(sharp_frame))
    colour = feed.get_colour_frame()
    edge_pixels = colour[:, 49:52, :]
    assert edge_pixels.min() == 0
    assert edge_pixels.max() == 255


def test_get_frame_rotates_180():
    rgb_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_frame[0, 0] = 255
    feed = Picamera2Feed(make_camera(rgb_frame))
    frame = feed.get_colour_frame()
    assert frame[99, 99].any()
    assert not frame[0, 0].any()