from time import monotonic

from gardendome.orchestrator.base import (
    AnnotatorMixin,
    BaseDisplayPipeline,
    FollowControllerMixin,
    MotionDetectorMixin,
    ObjectDetectorMixin,
    TurretMixin,
)


class BaseFollowPipeline(
    BaseDisplayPipeline, TurretMixin, AnnotatorMixin, FollowControllerMixin
):
    def __init__(self, config):
        super().__init__(config)
        self._setup_turret()
        self._setup_annotator()
        self._setup_follow_controller()

    def _follow(self, frame, bbox):
        centre = self._annotator.calculate_centre(bbox)
        frame = self._annotator.draw_centre(frame, centre)
        next_position = self._follow_controller.update(
            target_centre=centre,
            frame_shape=frame.shape,
            current=self._turret.position,
            now=monotonic(),
        )
        if next_position is not None:
            self._turret.move_to(next_position)
        return frame


class MotionFollowPipeline(BaseFollowPipeline, MotionDetectorMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_motion_detector()

    def _get_frame(self):
        frame, bbox = self._poll_motion_location()
        if bbox:
            frame = self._annotator.draw_box(frame, bbox)
            frame = self._follow(frame, bbox)
        return frame


class ObjectFollowPipeline(BaseFollowPipeline, ObjectDetectorMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_object_detector()

    def _get_frame(self):
        poll_start = monotonic()
        frame, target = self._poll_target()
        print(f"capture + detection: {monotonic() - poll_start:.3f}s")
        if target:
            frame = self._annotator.draw_detection(frame, target)
            frame = self._follow(frame, target.bbox)
        return frame
