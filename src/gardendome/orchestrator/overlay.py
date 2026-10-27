from gardendome.orchestrator.base import (
    BaseDisplayPipeline,
    MotionDetectorMixin,
    ObjectDetectorMixin,
    TrackerMixin,
)


class RawFeedPipeline(BaseDisplayPipeline):
    def __init__(self, config, colour=True):
        super().__init__(config)
        self._colour = colour

    def _get_frame(self):
        return (
            self._feed.get_colour_frame()
            if self._colour
            else self._feed.get_grey_frame()
        )


class MotionOverlayPipeline(BaseDisplayPipeline, MotionDetectorMixin, TrackerMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_motion_detector()
        self._setup_tracker()

    def _get_frame(self):
        frame, bbox = self._poll_motion_location()
        return self._tracker.draw_box(frame, bbox) if bbox else frame


class ObjectOverlayPipeline(BaseDisplayPipeline, ObjectDetectorMixin, TrackerMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_object_detector()
        self._setup_tracker()

    def _get_frame(self):
        frame, detections = self._poll_detections()
        for detection in detections:
            frame = self._tracker.draw_box(frame, detection.bbox)
            label = f"{detection.label} {detection.score:.0%}"
            frame = self._tracker.draw_label(frame, detection.bbox, label)
        return frame
