import pytest
from unittest.mock import MagicMock, patch
from camera.recorder import Picamera2Recorder


@pytest.fixture
def mock_camera():
    return MagicMock()


@pytest.fixture
def mock_logger():
    return MagicMock()


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


@patch("camera.recorder.FileOutput")
@patch("camera.recorder.datetime")
def test_start_recording(
    mock_datetime, mock_file_output, picamera2_recorder, mock_camera, mock_logger
):
    mock_datetime.now.return_value.strftime.return_value = "20250101_120000"

    picamera2_recorder.start(prefix="test")

    expected_filename = "test_20250101_120000.h264"
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


def test_start_while_already_recording(picamera2_recorder):
    picamera2_recorder._is_recording = True

    with pytest.raises(RuntimeError) as exc:
        picamera2_recorder.start(prefix="test")

    assert "Cannot start recording: Recorder is already active." in str(exc.value)


def test_stop_recording(picamera2_recorder, mock_camera, mock_logger):
    picamera2_recorder._is_recording = True
    picamera2_recorder._current_file = "test_prefix_20230101_120000.h264"
    picamera2_recorder.stop()

    assert picamera2_recorder.is_recording is False
    assert picamera2_recorder._current_file is None

    mock_camera.stop_recording.assert_called_once()
    mock_logger.video.assert_called_once_with(
        "/tmp/",
        "test_prefix_20230101_120000.h264",
    )

    mock_logger.info.assert_called_once_with("Recording stopped.")


def test_stop_when_not_recording(picamera2_recorder):
    picamera2_recorder._is_recording = False

    with pytest.raises(RuntimeError) as exc:
        picamera2_recorder.stop()

    assert "Cannot stop recording: Recorder is not active." in str(exc.value)
