import argparse

from gardendome import config
from gardendome.orchestrator.live_feed import LiveFeedPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grey', action='store_true', help='Display greyscale feed')
    args = parser.parse_args()

    with LiveFeedPipeline(config, colour=not args.grey) as pipeline:
        pipeline.run()