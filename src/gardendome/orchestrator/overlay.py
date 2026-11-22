from gardendome.orchestrator.base import (
    AnnotatorMixin,
    BaseDisplayPipeline,
    MotionDetectorMixin,
    ObjectDetectorMixin,
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


class MotionOverlayPipeline(BaseDisplayPipeline, MotionDetectorMixin, AnnotatorMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_motion_detector()
        self._setup_annotator()

    def _get_frame(self):
        frame, bbox = self._poll_motion_location()
        return self._annotator.draw_box(frame, bbox) if bbox else frame


class ObjectOverlayPipeline(BaseDisplayPipeline, ObjectDetectorMixin, AnnotatorMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_object_detector()
        self._setup_annotator()

    def _get_frame(self):
        frame, detections = self._poll_detections()
        for detection in detections:
            frame = self._annotator.draw_detection(frame, detection)
        return frame
