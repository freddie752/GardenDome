from gardendome.orchestrator.turret import TurretDemoPipeline

from gardendome import config

if __name__ == "__main__":
    with TurretDemoPipeline(config) as pipeline:
        pipeline.run()
