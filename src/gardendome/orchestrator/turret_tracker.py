from gardendome.orchestrator.base import (
    BaseDisplayPipeline,
    MotionDetectorMixin,
    TrackerMixin,
    TurretMixin,
)


class TurretTracker(
    BaseDisplayPipeline, MotionDetectorMixin, TrackerMixin, TurretMixin
):
    def __init__(self, config):
        super().__init__(config)
        self._setup_logging()
        self._setup_turret()
        self._setup_motion_detector()
        self._setup_tracker()
        self._settle_counter = 0

    def _get_frame(self):
        frame, bbox = self._get_motion_bbox()
        if self._settle_counter > 0:
            self._settle_counter -= 1
        elif bbox:
            frame = self._tracker.draw_box(frame, bbox)
            centre = self._tracker.calculate_centre(bbox)
            frame = self._tracker.draw_centre(frame, centre)
            self._turret.aim(centre, frame.shape)
            if self._turret.moved:
                self._settle_counter = self._config.TURRET_SETTLE_FRAMES
        return frame
