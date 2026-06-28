from gardendome.orchestrator.base import BaseDisplayPipeline, MotionDetectorMixin, TrackerMixin


class LiveFeedPipeline(BaseDisplayPipeline):
    def __init__(self, config, colour=True):
        super().__init__(config)
        self._colour = colour

    def _get_frame(self):
        return (
            self._feed.get_colour_frame()
            if self._colour
            else self._feed.get_grey_frame()
        )


class MotionDisplayPipeline(BaseDisplayPipeline, MotionDetectorMixin, TrackerMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_motion_detector()
        self._setup_tracker()

    def _get_frame(self):
        frame, bbox = self._get_motion_bbox()
        return self._tracker.draw_box(frame, bbox) if bbox else frame
