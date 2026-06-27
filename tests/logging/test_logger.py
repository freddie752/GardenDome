import pytest
from unittest.mock import MagicMock
 
from notifications.logger import Logger
 
 
@pytest.fixture
def mock_bot():
    return MagicMock()

def test_info_prints_message(capsys, mock_bot):
    log = Logger(slack_logging=False, slack_bot=mock_bot)
    log.info("Service started successfully")
    assert "Service started successfully" in capsys.readouterr().out

 
def test_info_sends_to_slack_when_enabled(mock_bot):
    log = Logger(slack_logging=True, slack_bot=mock_bot)
    log.info("Initialising recording pipeline")
    mock_bot.send_message.assert_called_once_with("Initialising recording pipeline")
 

def test_info_skips_slack_when_disabled(mock_bot):
    log = Logger(slack_logging=False, slack_bot=mock_bot)
    log.info("Initialising recording pipeline")
    mock_bot.send_message.assert_not_called()
 

def test_info_raises_when_slack_enabled_but_no_bot():
    log = Logger(slack_logging=True, slack_bot=None)
    with pytest.raises(AttributeError):
        log.info("Motion detected")
 
 
def test_video_sends_to_slack_when_enabled(mock_bot):
    log = Logger(slack_logging=True, slack_bot=mock_bot)
    log.video("/recordings/2024-01-15/", "motion_event.mp4")
    mock_bot.send_video.assert_called_once_with("/recordings/2024-01-15/", "motion_event.mp4")
 
 
def test_video_skips_slack_when_disabled(mock_bot):
    log = Logger(slack_logging=False, slack_bot=mock_bot)
    log.video("/recordings/2024-01-15/", "motion_event.mp4")
    mock_bot.send_video.assert_not_called()
 
 
# BUG: video() builds its print path as f"{video_dir}{video_filename}" with no
# separator, so Logger.video("/recordings", "motion_event.mp4") prints
# "/recordingsmotion_event.mp4" instead of "/recordings/motion_event.mp4".
def test_video_print_contains_valid_path(capsys, mock_bot):
    log = Logger(slack_logging=False, slack_bot=mock_bot)
    log.video("/recordings", "motion_event.mp4")  # no trailing slash
    out = capsys.readouterr().out
    assert "/recordings/motion_event.mp4" in out or "/recordings\\motion_event.mp4" in out, (
        f"Expected a properly joined path in log output, got: {out!r}"
    )