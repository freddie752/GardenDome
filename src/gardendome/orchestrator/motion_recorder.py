"""Orchestrator that wires capture, detection, tracking, recording, and notification."""

from gardendome.orchestrator.base import (
    BasePipeline,
    LoggingMixin,
    CameraMixin,
    RecorderMixin,
    MotionDetectorMixin,
)


class MotionRecordingPipeline(
    BasePipeline, LoggingMixin, CameraMixin, RecorderMixin, MotionDetectorMixin
):
    def __init__(self, config):
        super().__init__(config)
        self._setup_logging()
        self._setup_camera()
        self._setup_recorder()
        self._setup_motion_detector()
        self._previous_motion = False

    def run(self):
        self._logger.info("Starting motion detection and recording.")
        while True:
            self._step()

    def stop(self):
        self._logger.info("Stopping motion detection and recording..")
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
        current_motion = self._get_motion()
        self._handle_transitions(current_motion)
        self._previous_motion = current_motion
