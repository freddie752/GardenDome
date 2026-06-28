import cv2
from gardendome.orchestrator.base import (
    BasePipeline,
    LoggingMixin,
    CameraMixin,
)


class LiveFeedPipeline(BasePipeline, CameraMixin, LoggingMixin):
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

    def _step(self):
        frame = self._feed.get_colour_frame()
        cv2.imshow("Live Feed", frame)
        return cv2.waitKey(1) & 0xFF != ord('q')