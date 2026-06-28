from unittest.mock import MagicMock
import numpy as np

from gardendome.orchestrator.motion_recorder import MotionRecordingPipeline

def make_pipeline():
    pipeline = MotionRecordingPipeline.__new__(MotionRecordingPipeline)
    pipeline._config = MagicMock()
    pipeline._logger = MagicMock()
    pipeline._feed = MagicMock()
    pipeline._recorder = MagicMock()
    pipeline._motion_detector = MagicMock()
    pipeline._picam2 = MagicMock()
    pipeline._previous_frame = np.zeros((100, 100), dtype=np.uint8)
    pipeline._previous_motion = False
    return pipeline

def test_motion_start_triggers_recorder_start():
    pipeline = make_pipeline()
    pipeline._handle_transitions(current_motion=True)
    pipeline._recorder.start.assert_called_once_with(prefix="motion")


def test_motion_start_logs_detection():
    pipeline = make_pipeline()
    pipeline._handle_transitions(current_motion=True)
    pipeline._logger.info.assert_called_once()


def test_motion_stop_triggers_recorder_stop():
    pipeline = make_pipeline()
    pipeline._previous_motion = True
    pipeline._handle_transitions(current_motion=False)
    pipeline._recorder.stop.assert_called_once()


def test_motion_stop_logs_stopped():
    pipeline = make_pipeline()
    pipeline._previous_motion = True
    pipeline._handle_transitions(current_motion=False)
    pipeline._logger.info.assert_called_once()


def test_no_action_when_motion_continues():
    pipeline = make_pipeline()
    pipeline._previous_motion = True
    pipeline._handle_transitions(current_motion=True)
    pipeline._recorder.start.assert_not_called()
    pipeline._recorder.stop.assert_not_called()


def test_no_action_when_no_motion_continues():
    pipeline = make_pipeline()
    pipeline._previous_motion = False
    pipeline._handle_transitions(current_motion=False)
    pipeline._recorder.start.assert_not_called()
    pipeline._recorder.stop.assert_not_called()


def test_step_updates_previous_frame():
    pipeline = make_pipeline()
    new_frame = np.ones((100, 100), dtype=np.uint8) * 128
    pipeline._feed.get_frame.return_value = new_frame
    pipeline._motion_detector.detect.return_value = False

    pipeline._step()

    assert pipeline._previous_frame is new_frame


def test_step_updates_previous_motion():
    pipeline = make_pipeline()
    pipeline._feed.get_frame.return_value = np.zeros((100, 100), dtype=np.uint8)
    pipeline._motion_detector.detect.return_value = True

    pipeline._step()

    assert pipeline._previous_motion is True


def test_step_passes_correct_frames_to_detector():
    pipeline = make_pipeline()
    previous_frame = pipeline._previous_frame
    new_frame = np.ones((100, 100), dtype=np.uint8) * 64
    pipeline._feed.get_frame.return_value = new_frame
    pipeline._motion_detector.detect.return_value = False

    pipeline._step()

    pipeline._motion_detector.detect.assert_called_once_with(
        current_frame=new_frame,
        previous_frame=previous_frame,
    )


def test_stop_calls_picam2_stop_and_close():
    pipeline = make_pipeline()
    pipeline.stop()
    pipeline._picam2.stop.assert_called_once()
    pipeline._picam2.close.assert_called_once()