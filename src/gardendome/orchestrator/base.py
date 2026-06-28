from abc import ABC, abstractmethod
from gardendome.notifications.logger import Logger
from gardendome.notifications.slack import SlackBot
from picamera2 import Picamera2
from gardendome.camera.feed import Picamera2Feed
from gardendome.camera.recorder import Picamera2Recorder
from gardendome.detectors.motion_detector import MotionDetector
from gardendome.turret_control.turret import Turret


class BasePipeline(ABC):
    def __init__(self, config):
        self._config = config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    @abstractmethod
    def stop():
        pass

    @abstractmethod
    def run():
        pass


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
        self._previous_frame = self._feed.get_grey_frame()

    def _get_motion(self):
        current_frame = self._feed.get_grey_frame()

        current_motion = self._motion_detector.has_motion(
            current_frame=current_frame,
            previous_frame=self._previous_frame,
        )
        self._previous_frame = current_frame
        return current_motion


class TurretMixin:
    def _setup_turret(self):
        self._turret = Turret(
            logger=self._logger,
            min_tilt=self._config.TURRET_MIN_TILT,
            max_tilt=self._config.TURRET_MAX_TILT,
            min_pan=self._config.TURRET_MIN_PAN,
            max_pan=self._config.TURRET_MAX_PAN,
            start_tilt=self._config.TURRET_START_TILT,
            start_pan=self._config.TURRET_START_PAN,
        )
