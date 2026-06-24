"""Orchestrator that wires capture, detection, tracking, recording, and notification."""

from picamera2 import Picamera2
from camera.feed import Picamera2Feed
from camera.recorder import Picamera2Recorder
from logging.logger import Logger
from logging.slack import SlackBot
from detectors.motion_detector import MotionDetector


class MotionRecordingPipeline:
    def __init__(self, config):
        self._config = config
        self._setup_logging()
        self._setup_camera_components()
        self._previous_motion = False
        self._previous_frame = self._feed.get_frame()

    def _setup_logging(self):
        if self._config.SLACK_LOGGING:
            slack_bot = SlackBot(slack_channel=self._config.SLACK_CHANNEL)
            self._logger = Logger(slack_logging=True, slack_bot=slack_bot)
        else:
            self._logger = Logger(slack_logging=False)

    def _setup_camera_components(self):
        self._picam2 = Picamera2()
        camera_config = self._picam2.create_video_configuration()
        self._picam2.configure(camera_config)
        self._picam2.start()
        self._feed = Picamera2Feed(camera=self._picam2)
        self._recorder = Picamera2Recorder(
            logger=self._logger,
            camera=self._picam2,
            recording_dir=self._config.RECORDING_DIR,
            bitrate=self._config.BITRATE,
        )
        self._motion_detector = MotionDetector(
            motion_threshold=self._config.MOTION_THRESHOLD,
            motion_fraction=self._config.MOTION_FRACTION,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def run(self):
        self._logger.info("Starting motion detection and recording.")
        while True:
            self._step()

    def stop(self):
        self._logger.info("Stopping pipeline.")
        self._picam2.stop()
        self._picam2.close()

    def _handle_transitions(self, current_motion):
        if current_motion and not self._previous_motion:
            self._on_motion_start()

        elif not current_motion and self._previous_motion:
            self._on_motion_stop()

    def _on_motion_start(self):
        self._logger.info("Motion detected.")
        self._recorder.start(prefix="motion")

    def _on_motion_stop(self):
        self._logger.info("Motion stopped.")
        self._recorder.stop()

    def _step(self):
        current_frame = self._feed.get_frame()

        current_motion = self._motion_detector.detect(
            current_frame=current_frame,
            previous_frame=self._previous_frame,
        )

        self._handle_transitions(current_motion)

        self._previous_frame = current_frame
        self._previous_motion = current_motion
