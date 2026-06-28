import numpy as np
from unittest.mock import MagicMock, patch
from gardendome.orchestrator.live_feed import LiveFeedPipeline


def make_pipeline():
    pipeline = LiveFeedPipeline.__new__(LiveFeedPipeline)
    pipeline._config = MagicMock()
    pipeline._logger = MagicMock()
    pipeline._feed = MagicMock()
    pipeline._picam2 = MagicMock()
    return pipeline


def test_step_returns_true_when_q_not_pressed():
    pipeline = make_pipeline()
    pipeline._feed.get_colour_frame.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    with (
        patch("gardendome.orchestrator.live_feed.cv2.imshow"),
        patch("gardendome.orchestrator.live_feed.cv2.waitKey", return_value=0),
    ):
        assert pipeline._step() is True


def test_step_returns_false_when_q_pressed():
    pipeline = make_pipeline()
    pipeline._feed.get_colour_frame.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    with (
        patch("gardendome.orchestrator.live_feed.cv2.imshow"),
        patch("gardendome.orchestrator.live_feed.cv2.waitKey", return_value=ord("q")),
    ):
        assert pipeline._step() is False


def test_step_gets_colour_frame():
    pipeline = make_pipeline()
    pipeline._feed.get_colour_frame.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    with (
        patch("gardendome.orchestrator.live_feed.cv2.imshow"),
        patch("gardendome.orchestrator.live_feed.cv2.waitKey", return_value=0),
    ):
        pipeline._step()
    pipeline._feed.get_colour_frame.assert_called_once()


def test_step_shows_frame_in_correct_window():
    pipeline = make_pipeline()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline._feed.get_colour_frame.return_value = frame
    with (
        patch("gardendome.orchestrator.live_feed.cv2.imshow") as mock_imshow,
        patch("gardendome.orchestrator.live_feed.cv2.waitKey", return_value=0),
    ):
        pipeline._step()
    mock_imshow.assert_called_once_with("Live Feed", frame)


def test_stop_cleans_up():
    pipeline = make_pipeline()
    with patch(
        "gardendome.orchestrator.live_feed.cv2.destroyAllWindows"
    ) as mock_destroy:
        pipeline.stop()
    mock_destroy.assert_called_once()
    pipeline._picam2.stop.assert_called_once()
    pipeline._picam2.close.assert_called_once()
    pipeline._logger.info.assert_called_once()
