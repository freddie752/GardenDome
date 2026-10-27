from gardendome import config
from gardendome.orchestrator.turret_follow import MotionFollowPipeline

if __name__ == '__main__':
    with MotionFollowPipeline(config) as pipeline:
        pipeline.run()