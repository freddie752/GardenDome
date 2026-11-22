import numpy as np

from gardendome.annotation.annotator import Annotator


def grey_frame(height=100, width=100):
    return np.zeros((height, width), dtype=np.uint8)


def test_calculate_centre_returns_correct_coords():
    annotator = Annotator()
    assert annotator.calculate_centre((10, 20, 40, 60)) == (30, 50)


def test_calculate_centre_returns_integers():
    annotator = Annotator()
    cx, cy = annotator.calculate_centre((10, 20, 41, 61))
    assert isinstance(cx, int)
    assert isinstance(cy, int)


def test_draw_box_modifies_frame():
    annotator = Annotator()
    frame = grey_frame()
    result = annotator.draw_box(frame, (10, 10, 30, 30))
    assert not np.array_equal(result, grey_frame())


def test_draw_centre_modifies_frame():
    annotator = Annotator()
    frame = grey_frame()
    result = annotator.draw_centre(frame, (50, 50))
    assert not np.array_equal(result, grey_frame())
