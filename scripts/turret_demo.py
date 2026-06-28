from gardendome import config
from gardendome.orchestrator.turret import TurretDemoPipeline

if __name__ == '__main__':
    with TurretDemoPipeline(config) as pipeline:
            pipeline.run()