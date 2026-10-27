from gardendome import config
from gardendome.orchestrator.overlay import MotionOverlayPipeline

if __name__ == '__main__':
    with MotionOverlayPipeline(config) as pipeline:
        pipeline.run()