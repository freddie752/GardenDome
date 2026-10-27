from gardendome import config
from gardendome.orchestrator.turret_follow import ObjectFollowPipeline

if __name__ == '__main__':
    with ObjectFollowPipeline(config) as pipeline:
        pipeline.run()