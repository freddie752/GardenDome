from notifications.logger import Logger
from notifications.slack import SlackBot
from picamera2 import Picamera2
from camera.feed import Picamera2Feed
from camera.recorder import Picamera2Recorder
from detectors.motion_detector import MotionDetector


class BasePipeline:
    def __init__(self, config):
        self._config = config


class LoggingMixin:
    def _setup_logging(self):
        if self._config.SLACK_LOGGING:
            slack_bot = SlackBot(slack_channel=self._config.SLACK_CHANNEL)
            self._logger = Logger(slack_logging=True, slack_bot=slack_bot)
        else:
            self._logger = Logger(slack_logging=False)


class CameraMixin:
    def _setup_camera(self):
        self._picam2 = Picamera2()
        camera_config = self._picam2.create_video_configuration()
        self._picam2.configure(camera_config)
        self._picam2.start()
        self._feed = Picamera2Feed(camera=self._picam2)


class RecorderMixin:
    def _setup_recorder(self):
        self._recorder = Picamera2Recorder(
            logger=self._logger,
            camera=self._picam2,
            recording_dir=self._config.RECORDING_DIR,
            bitrate=self._config.BITRATE,
        )


class MotionDetectorMixin:
    def _setup_motion_detector(self):
        self._motion_detector = MotionDetector(
            motion_threshold=self._config.MOTION_THRESHOLD,
            motion_fraction=self._config.MOTION_FRACTION,
        )
