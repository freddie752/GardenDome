from gardendome import config
from gardendome.orchestrator.turret_tracker import TurretTracker

if __name__ == '__main__':
    with TurretTracker(config) as pipeline:
        pipeline.run()