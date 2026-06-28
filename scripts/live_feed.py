from gardendome import config
from gardendome.orchestrator.live_feed import LiveFeedPipeline

if __name__ == '__main__':
    with LiveFeedPipeline(config) as pipeline:
            pipeline.run()