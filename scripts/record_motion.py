from gardendome import config
from gardendome.orchestrator.motion_recorder import MotionRecordingPipeline

if __name__ == '__main__':
    with MotionRecordingPipeline(config) as pipeline:
            pipeline.run()