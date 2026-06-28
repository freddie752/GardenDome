import cv2
from abc import ABC, abstractmethod
from gardendome.notifications.logger import Logger
from gardendome.notifications.slack import SlackBot
from picamera2 import Picamera2
from gardendome.camera.feed import Picamera2Feed
from gardendome.camera.recorder import Picamera2Recorder
from gardendome.detectors.motion_detector import MotionDetector
from gardendome.turret_control.turret import Turret
from gardendome.tracking.tracker import Tracker


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

    def _get_motion_bbox(self):
        current_frame = self._feed.get_grey_frame()
        bbox = self._motion_detector.locate_motion(
            current_frame=current_frame,
            previous_frame=self._previous_frame,
        )
        self._previous_frame = current_frame
        return current_frame, bbox


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


class TrackerMixin:
    def _setup_tracker(self):
        self._tracker = Tracker()


class BaseDisplayPipeline(BasePipeline, CameraMixin, LoggingMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_logging()
        self._setup_camera()
        cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Feed", 640, 480)

    def run(self):
        self._logger.info("Starting live feed.")
        while self._step():
            pass

    def stop(self):
        self._logger.info("Stopping live feed.")
        cv2.destroyAllWindows()
        self._picam2.stop()
        self._picam2.close()

    @abstractmethod
    def _get_frame(self):
        pass

    def _step(self):
        frame = self._get_frame()
        cv2.imshow("Live Feed", frame)
        return cv2.waitKey(1) & 0xFF != ord("q")
