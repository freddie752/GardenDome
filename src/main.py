from detect_motion_record import MotionDetectorRecorder
from slack import SlackBot
from config import RECORDING_DIR, BITRATE, MOTION_THRESHOLD
import os
import threading

SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

if __name__ == '__main__':
    slack_bot = SlackBot()
    motion_detector_recorder = MotionDetectorRecorder(RECORDING_DIR, BITRATE, MOTION_THRESHOLD, slack_bot=slack_bot)
    slack_thread = threading.Thread(target=slack_bot.start, daemon=True)
    motion_thread = threading.Thread(target=motion_detector_recorder.detect_motion_record, daemon=True)

    slack_thread.start()
    motion_thread.start()

    slack_thread.join()
    motion_thread.join()
    
