from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np

from gardendome.annotation.annotator import Annotator
from gardendome.detectors.object_detector import Detection
from gardendome.orchestrator.base import BaseDisplayPipeline
from gardendome.orchestrator.turret_follow import (
    MotionFollowPipeline,
    ObjectFollowPipeline,
)
from gardendome.turret_control.angles import Angles

BBOX = (20, 30, 40, 50)
BBOX_CENTRE = (40, 55)
NEXT_POSITION = Angles(pan=80, tilt=110)


def make_pipeline(pipeline_class):
    pipeline = pipeline_class.__new__(pipeline_class)
    pipeline._feed = MagicMock()
    pipeline._annotator = Annotator()
    pipeline._turret = MagicMock()
    pipeline._turret.position = Angles(pan=90, tilt=120)
    pipeline._follow_controller = MagicMock()
    pipeline._follow_controller.update.return_value = NEXT_POSITION
    return pipeline


def colour_frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_follow_moves_turret_to_controller_result():
    pipeline = make_pipeline(ObjectFollowPipeline)
    pipeline._follow(colour_frame(), BBOX)
    pipeline._turret.move_to.assert_called_once_with(NEXT_POSITION)


def test_follow_does_not_move_turret_when_controller_returns_none():
    pipeline = make_pipeline(ObjectFollowPipeline)
    pipeline._follow_controller.update.return_value = None
    pipeline._follow(colour_frame(), BBOX)
    pipeline._turret.move_to.assert_not_called()


def test_follow_passes_centre_shape_position_and_time_to_controller():
    pipeline = make_pipeline(ObjectFollowPipeline)
    with patch("gardendome.orchestrator.turret_follow.monotonic", return_value=12.5):
        pipeline._follow(colour_frame(), BBOX)
    pipeline._follow_controller.update.assert_called_once_with(
        target_centre=BBOX_CENTRE,
        frame_shape=(200, 200, 3),
        current=Angles(pan=90, tilt=120),
        now=12.5,
    )


def test_follow_draws_centre_on_frame():
    pipeline = make_pipeline(ObjectFollowPipeline)
    frame = pipeline._follow(colour_frame(), BBOX)
    assert tuple(frame[55, 40]) == (255, 255, 255)


def test_object_follow_follows_target():
    pipeline = make_pipeline(ObjectFollowPipeline)
    pipeline._object_detector = MagicMock()
    new_frame = colour_frame()
    pipeline._feed.get_colour_frame.return_value = new_frame
    pipeline._object_detector.detect_target.return_value = Detection(
        label="person", score=0.87, bbox=BBOX
    )

    frame = pipeline._get_frame()

    assert frame is new_frame
    assert tuple(frame[30, 20]) == (255, 255, 255)
    pipeline._turret.move_to.assert_called_once_with(NEXT_POSITION)


def test_object_follow_without_target_does_not_update_controller():
    pipeline = make_pipeline(ObjectFollowPipeline)
    pipeline._object_detector = MagicMock()
    pipeline._feed.get_colour_frame.return_value = colour_frame()
    pipeline._object_detector.detect_target.return_value = None

    frame = pipeline._get_frame()

    assert not frame.any()
    pipeline._follow_controller.update.assert_not_called()


def test_motion_follow_follows_motion():
    pipeline = make_pipeline(MotionFollowPipeline)
    pipeline._motion_detector = MagicMock()
    pipeline._previous_frame = np.zeros((200, 200), dtype=np.uint8)
    new_frame = np.zeros((200, 200), dtype=np.uint8)
    pipeline._feed.get_grey_frame.return_value = new_frame
    pipeline._motion_detector.locate_motion.return_value = BBOX

    frame = pipeline._get_frame()

    assert frame is new_frame
    assert frame[30, 20] == 255
    pipeline._turret.move_to.assert_called_once_with(NEXT_POSITION)


def test_motion_follow_without_motion_does_not_update_controller():
    pipeline = make_pipeline(MotionFollowPipeline)
    pipeline._motion_detector = MagicMock()
    pipeline._previous_frame = np.zeros((200, 200), dtype=np.uint8)
    pipeline._feed.get_grey_frame.return_value = np.zeros((200, 200), dtype=np.uint8)
    pipeline._motion_detector.locate_motion.return_value = None

    frame = pipeline._get_frame()

    assert not frame.any()
    pipeline._follow_controller.update.assert_not_called()


def test_object_follow_init_runs_all_setups():
    with (
        patch.object(BaseDisplayPipeline, "__init__", return_value=None),
        patch.multiple(
            ObjectFollowPipeline,
            _setup_turret=DEFAULT,
            _setup_annotator=DEFAULT,
            _setup_follow_controller=DEFAULT,
            _setup_object_detector=DEFAULT,
        ) as setups,
    ):
        ObjectFollowPipeline(MagicMock())
    for setup in setups.values():
        setup.assert_called_once()


def test_motion_follow_init_runs_all_setups():
    with (
        patch.object(BaseDisplayPipeline, "__init__", return_value=None),
        patch.multiple(
            MotionFollowPipeline,
            _setup_turret=DEFAULT,
            _setup_annotator=DEFAULT,
            _setup_follow_controller=DEFAULT,
            _setup_motion_detector=DEFAULT,
        ) as setups,
    ):
        MotionFollowPipeline(MagicMock())
    for setup in setups.values():
        setup.assert_called_once()
