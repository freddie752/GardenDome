import numpy as np
from gardendome.tracking.tracker import Tracker


def grey_frame(height=100, width=100):
    return np.zeros((height, width), dtype=np.uint8)


def test_calculate_centre_returns_correct_coords():
    tracker = Tracker()
    assert tracker.calculate_centre((10, 20, 40, 60)) == (30, 50)


def test_calculate_centre_returns_integers():
    tracker = Tracker()
    cx, cy = tracker.calculate_centre((10, 20, 41, 61))
    assert isinstance(cx, int)
    assert isinstance(cy, int)


def test_draw_box_modifies_frame():
    tracker = Tracker()
    frame = grey_frame()
    result = tracker.draw_box(frame, (10, 10, 30, 30))
    assert not np.array_equal(result, grey_frame())


def test_draw_centre_modifies_frame():
    tracker = Tracker()
    frame = grey_frame()
    result = tracker.draw_centre(frame, (50, 50))
    assert not np.array_equal(result, grey_frame())
