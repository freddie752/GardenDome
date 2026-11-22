from gardendome.orchestrator.base import (
    AnnotatorMixin,
    BaseDisplayPipeline,
    MotionDetectorMixin,
    ObjectDetectorMixin,
    TurretMixin,
)


class MotionFollowPipeline(
    BaseDisplayPipeline, MotionDetectorMixin, AnnotatorMixin, TurretMixin
):
    def __init__(self, config):
        super().__init__(config)
        self._setup_turret()
        self._setup_motion_detector()
        self._setup_annotator()
        self._settle_counter = 0

    def _get_frame(self):
        frame, bbox = self._poll_motion_location()
        if self._settle_counter > 0:
            self._settle_counter -= 1
        elif bbox:
            frame = self._annotator.draw_box(frame, bbox)
            centre = self._annotator.calculate_centre(bbox)
            frame = self._annotator.draw_centre(frame, centre)
            self._turret.aim(centre, frame.shape)
            if self._turret.moved:
                self._settle_counter = self._config.TURRET_SETTLE_FRAMES
        return frame


class ObjectFollowPipeline(
    BaseDisplayPipeline, ObjectDetectorMixin, AnnotatorMixin, TurretMixin
):
    def __init__(self, config):
        super().__init__(config)
        self._setup_turret()
        self._setup_object_detector()
        self._setup_annotator()
        self._settle_counter = 0

    def _get_frame(self):
        frame, target = self._poll_target()
        if target:
            frame = self._annotator.draw_detection(frame, target)
            centre = self._annotator.calculate_centre(target.bbox)
            frame = self._annotator.draw_centre(frame, centre)
            self._turret.aim(centre, frame.shape)
        return frame
