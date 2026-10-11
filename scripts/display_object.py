from gardendome import config
from gardendome.orchestrator.live_feed import ObjectDisplayPipeline

if __name__ == '__main__':
    with ObjectDisplayPipeline(config) as pipeline:
        pipeline.run()