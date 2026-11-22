from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gardendome import config
from gardendome.annotation.annotator import Annotator
from gardendome.camera.feed import Picamera2Feed
from gardendome.camera.recorder import Picamera2Recorder
from gardendome.detectors.motion_detector import MotionDetector
from gardendome.notifications.logger import Logger
from gardendome.orchestrator.base import (
    AnnotatorMixin,
    CameraMixin,
    FollowControllerMixin,
    LoggingMixin,
    MotionDetectorMixin,
    ObjectDetectorMixin,
    RecorderMixin,
    TurretMixin,
)
from gardendome.orchestrator.motion_recorder import MotionRecordingPipeline
from gardendome.orchestrator.overlay import (
    MotionOverlayPipeline,
    ObjectOverlayPipeline,
    RawFeedPipeline,
)
from gardendome.orchestrator.turret_demo import TurretDemoPipeline
from gardendome.orchestrator.turret_follow import (
    MotionFollowPipeline,
    ObjectFollowPipeline,
)
from gardendome.turret_control.angles import Angles
from gardendome.turret_control.follow_controller import FollowController


def make_mixin(mixin_class, **attributes):
    mixin = mixin_class()
    mixin._config = config
    for name, value in attributes.items():
        setattr(mixin, name, value)
    return mixin


@pytest.fixture
def fake_hardware():
    with (
        patch("gardendome.orchestrator.base.SlackBot"),
        patch("gardendome.orchestrator.base.Picamera2") as mock_picamera2,
        patch("gardendome.orchestrator.base.ObjectDetector"),
        patch("gardendome.orchestrator.base.cv2.namedWindow"),
        patch("gardendome.orchestrator.base.cv2.resizeWindow"),
    ):
        mock_picamera2.return_value.capture_array.return_value = np.zeros(
            (480, 640, 3), dtype=np.uint8
        )
        yield


def test_setup_logging_creates_logger():
    mixin = make_mixin(LoggingMixin)
    with patch("gardendome.orchestrator.base.SlackBot"):
        mixin._setup_logging()
    assert isinstance(mixin._logger, Logger)


def test_setup_camera_configures_camera_without_frame_queue():
    mixin = make_mixin(CameraMixin)
    with patch("gardendome.orchestrator.base.Picamera2") as mock_picamera2:
        mixin._setup_camera()
    camera = mock_picamera2.return_value
    camera.create_video_configuration.assert_called_once_with(queue=False)
    camera.start.assert_called_once()
    assert isinstance(mixin._feed, Picamera2Feed)


def test_setup_recorder_creates_recorder():
    mixin = make_mixin(RecorderMixin, _logger=MagicMock(), _picam2=MagicMock())
    mixin._setup_recorder()
    assert isinstance(mixin._recorder, Picamera2Recorder)


def test_setup_motion_detector_stores_first_grey_frame():
    feed = MagicMock()
    mixin = make_mixin(MotionDetectorMixin, _feed=feed)
    mixin._setup_motion_detector()
    assert isinstance(mixin._motion_detector, MotionDetector)
    assert mixin._previous_frame is feed.get_grey_frame.return_value


def test_setup_object_detector_creates_detector():
    mixin = make_mixin(ObjectDetectorMixin)
    with patch("gardendome.orchestrator.base.ObjectDetector") as mock_detector:
        mixin._setup_object_detector()
    assert mixin._object_detector is mock_detector.return_value


def test_setup_turret_starts_at_configured_position():
    mixin = make_mixin(TurretMixin, _logger=MagicMock())
    mixin._setup_turret()
    assert mixin._turret.position == Angles(
        pan=config.TURRET_START_PAN, tilt=config.TURRET_START_TILT
    )


def test_setup_follow_controller_creates_controller():
    mixin = make_mixin(FollowControllerMixin)
    mixin._setup_follow_controller()
    assert isinstance(mixin._follow_controller, FollowController)


def test_setup_annotator_creates_annotator():
    mixin = make_mixin(AnnotatorMixin)
    mixin._setup_annotator()
    assert isinstance(mixin._annotator, Annotator)


@pytest.mark.parametrize(
    "pipeline_class",
    [
        RawFeedPipeline,
        MotionOverlayPipeline,
        ObjectOverlayPipeline,
        MotionFollowPipeline,
        ObjectFollowPipeline,
        MotionRecordingPipeline,
        TurretDemoPipeline,
    ],
)
def test_pipeline_constructs_with_real_config(fake_hardware, pipeline_class):
    pipeline = pipeline_class(config)
    assert pipeline._config is config
