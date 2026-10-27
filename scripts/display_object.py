from gardendome import config
from gardendome.orchestrator.overlay import ObjectOverlayPipeline

if __name__ == '__main__':
    with ObjectOverlayPipeline(config) as pipeline:
        pipeline.run()