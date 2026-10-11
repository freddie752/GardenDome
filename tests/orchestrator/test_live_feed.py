import numpy as np
from unittest.mock import MagicMock, patch
from gardendome.detectors.object_detector import Detection
from gardendome.orchestrator.live_feed import (
    LiveFeedPipeline,
    MotionDisplayPipeline,
    ObjectDisplayPipeline,
)
from gardendome.tracking.tracker import Tracker


def make_pipeline():
    pipeline = LiveFeedPipeline.__new__(LiveFeedPipeline)
    pipeline._config = MagicMock()
    pipeline._logger = MagicMock()
    pipeline._feed = MagicMock()
    pipeline._picam2 = MagicMock()
    pipeline._colour = True
    return pipeline


def test_step_returns_true_when_q_not_pressed():
    pipeline = make_pipeline()
    pipeline._feed.get_colour_frame.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    with (
        patch("gardendome.orchestrator.base.cv2.imshow"),
        patch("gardendome.orchestrator.base.cv2.waitKey", return_value=0),
    ):
        assert pipeline._step() is True


def test_step_returns_false_when_q_pressed():
    pipeline = make_pipeline()
    pipeline._feed.get_colour_frame.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    with (
        patch("gardendome.orchestrator.base.cv2.imshow"),
        patch("gardendome.orchestrator.base.cv2.waitKey", return_value=ord("q")),
    ):
        assert pipeline._step() is False


def test_step_gets_colour_frame():
    pipeline = make_pipeline()
    pipeline._feed.get_colour_frame.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    with (
        patch("gardendome.orchestrator.base.cv2.imshow"),
        patch("gardendome.orchestrator.base.cv2.waitKey", return_value=0),
    ):
        pipeline._step()
    pipeline._feed.get_colour_frame.assert_called_once()


def test_step_shows_frame_in_correct_window():
    pipeline = make_pipeline()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline._feed.get_colour_frame.return_value = frame
    with (
        patch("gardendome.orchestrator.base.cv2.imshow") as mock_imshow,
        patch("gardendome.orchestrator.base.cv2.waitKey", return_value=0),
    ):
        pipeline._step()
    mock_imshow.assert_called_once_with("Live Feed", frame)


def test_stop_cleans_up():
    pipeline = make_pipeline()
    with patch(
        "gardendome.orchestrator.base.cv2.destroyAllWindows"
    ) as mock_destroy:
        pipeline.stop()
    mock_destroy.assert_called_once()
    pipeline._picam2.stop.assert_called_once()
    pipeline._picam2.close.assert_called_once()
    pipeline._logger.info.assert_called_once()


def test_step_gets_grey_frame_when_colour_false():
    pipeline = make_pipeline()
    pipeline._colour = False
    pipeline._feed.get_grey_frame.return_value = np.zeros((100, 100), dtype=np.uint8)
    with (
        patch("gardendome.orchestrator.base.cv2.imshow"),
        patch("gardendome.orchestrator.base.cv2.waitKey", return_value=0),
    ):
        pipeline._step()
    pipeline._feed.get_grey_frame.assert_called_once()
    pipeline._feed.get_colour_frame.assert_not_called()


def test_motion_display_draws_box_on_analysed_frame():
    pipeline = MotionDisplayPipeline.__new__(MotionDisplayPipeline)
    pipeline._feed = MagicMock()
    pipeline._motion_detector = MagicMock()
    pipeline._tracker = Tracker()
    previous_frame = np.zeros((200, 200), dtype=np.uint8)
    new_frame = np.zeros((200, 200), dtype=np.uint8)
    pipeline._previous_frame = previous_frame
    pipeline._feed.get_grey_frame.return_value = new_frame
    pipeline._motion_detector.locate_motion.return_value = (20, 30, 40, 50)

    frame = pipeline._get_frame()

    pipeline._motion_detector.locate_motion.assert_called_once_with(
        previous_frame=previous_frame, current_frame=new_frame
    )
    assert frame is new_frame
    assert frame[30, 20] == 255  # top-left corner
    assert frame[80, 60] == 255  # bottom-right corner


def test_object_display_draws_box_on_analysed_frame():
    pipeline = ObjectDisplayPipeline.__new__(ObjectDisplayPipeline)
    pipeline._feed = MagicMock()
    pipeline._object_detector = MagicMock()
    pipeline._tracker = Tracker()
    new_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    pipeline._feed.get_colour_frame.return_value = new_frame
    pipeline._object_detector.detect.return_value = [
        Detection(label="person", score=0.87, bbox=(20, 30, 40, 50))
    ]

    frame = pipeline._get_frame()

    pipeline._object_detector.detect.assert_called_once_with(new_frame)
    assert frame is new_frame
    assert tuple(frame[30, 20]) == (255, 255, 255)  # top-left corner
    assert tuple(frame[80, 60]) == (255, 255, 255)  # bottom-right corner
