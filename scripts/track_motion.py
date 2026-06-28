from gardendome import config
from gardendome.orchestrator.live_feed import MotionDisplayPipeline

if __name__ == '__main__':
    with MotionDisplayPipeline(config) as pipeline:
        pipeline.run()