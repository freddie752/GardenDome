import config
from src.orchestrator.motion_recorder import MotionRecordingPipeline

if __name__ == '__main__':
    with MotionRecordingPipeline(config) as pipeline:
            pipeline.run()