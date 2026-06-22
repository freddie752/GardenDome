import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
from src.camera.recorder import Picamera2Recorder

@pytest.fixture
def mock_camera():
    return MagicMock()

@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = MagicMock()
    logger.video = MagicMock()
    return logger

@pytest.fixture
def picamera2_recorder(mock_camera, mock_logger):
    return Picamera2Recorder(
        camera=mock_camera,
        recording_dir="/tmp/",
        bitrate=123456,
        logger=mock_logger,
    )

def test_initial_state(picamera2_recorder):
    assert picamera2_recorder.is_recording is False
    assert picamera2_recorder._current_file is None

@patch("src.camera.recorder.FileOutput")
@patch("src.camera.recorder.datetime")
@patch("builtins.open", new_callable=mock_open)
def test_start_recording(mock_open_file, mock_datetime, mock_file_output, picamera2_recorder, mock_camera, mock_logger):
    mock_datetime.now.return_value.strftime.return_value = "motion_20250101_120000.h264"

    picamera2_recorder.start(prefix="motion")

    expected_filename = "motion_20250101_120000.h264"
    expected_path = f"/tmp/{expected_filename}"

    assert picamera2_recorder.is_recording is True
    assert picamera2_recorder._current_file == expected_filename

    mock_file_output.assert_called_once_with(expected_path)
    mock_camera.start_recording.assert_called_once_with(
        picamera2_recorder._encoder,
        mock_file_output.return_value,
    )

    mock_logger.info.assert_called_once_with(
        f"Recording started. Storing at {expected_filename}"
    )

def test_start_while_already_recording_raises(picamera2_recorder):
    picamera2_recorder._is_recording = True

    with pytest.raises(RuntimeError) as exc:
        picamera2_recorder.start(prefix="motion")

    assert "Recorder is already active" in str(exc.value)

@patch("src.camera.recorder.datetime")
@patch("builtins.open", new_callable=mock_open)
def test_stop_recording(mock_open_file, mock_datetime, picamera2_recorder, mock_camera, mock_logger):
    # Setup a recording to be active first (without calling the actual start method to avoid file ops)
    mock_datetime.now.return_value.strftime.return_value = "test_prefix_20230101_120000.h264"
    picamera2_recorder._is_recording = True
    picamera2_recorder._current_file = "test_prefix_20230101_120000.h264"

    picamera2_recorder.stop()

    mock_camera.stop_recording.assert_called_once()

    mock_logger.video.assert_called_once_with(
        "/tmp/",
        "test_prefix_20230101_120000.h264",
    )
    mock_logger.info.assert_called_once_with("Recording stopped.")
    assert picamera2_recorder.is_recording is False
    assert picamera2_recorder._current_file is None