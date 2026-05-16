import pytest
from unittest.mock import MagicMock, patch

from src.camera.recorder import Picamera2Recorder


@pytest.fixture
def mock_camera():
    return MagicMock()


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.log = MagicMock()
    logger.video = MagicMock()
    return logger


@pytest.fixture
def recorder(mock_camera, mock_logger):
    return Picamera2Recorder(
        camera=mock_camera,
        recording_dir="/tmp/",
        bitrate=123456,
        logger=mock_logger,
    )


def test_initial_state(recorder):
    assert recorder.is_recording is False
    assert recorder._current_file is None


@patch("src.camera.recorder.FileOutput")
@patch("src.camera.recorder.datetime")
def test_start_recording(mock_datetime, mock_file_output, recorder, mock_camera, mock_logger):
    mock_datetime.now.return_value.strftime.return_value = "motion_20250101_120000.h264"

    recorder.start(prefix="motion")

    expected_filename = "motion_20250101_120000.h264"
    expected_path = f"/tmp/{expected_filename}"

    assert recorder.is_recording is True
    assert recorder._current_file == expected_filename

    mock_file_output.assert_called_once_with(expected_path)
    mock_camera.start_recording.assert_called_once_with(
        recorder._encoder,
        mock_file_output.return_value,
    )

    mock_logger.log.assert_called_once_with(
        f"Motion detected. Recording to {expected_filename}"
    )


def test_start_while_already_recording_raises(recorder):
    recorder._is_recording = True

    with pytest.raises(RuntimeError) as exc:
        recorder.start(prefix="motion")

    assert "Recorder is already active" in str(exc.value)


def test_stop_recording(recorder, mock_camera, mock_logger):
    recorder._current_file = "motion_20250101_120000.h264"

    recorder.stop()

    mock_camera.stop_recording.assert_called_once()

    mock_logger.log.assert_called_once_with("Motion stopped.")
    mock_logger.video.assert_called_once_with(
        "/tmp/",
        "motion_20250101_120000.h264",
    )

    assert recorder._current_file is None