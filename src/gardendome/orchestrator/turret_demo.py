import time
from gardendome.orchestrator.base import BasePipeline, LoggingMixin, TurretMixin


class TurretDemoPipeline(BasePipeline, LoggingMixin, TurretMixin):
    def __init__(self, config):
        super().__init__(config)
        self._setup_logging()
        self._setup_turret()
        self._pan_dir = 1
        self._tilt_dir = 1

    def run(self):
        self._logger.info("Starting turret demo.")
        while self._step():
            pass

    def stop(self):
        self._logger.info("Stopping turret demo.")
        self._turret.set_pan(self._config.TURRET_START_PAN)
        self._turret.set_tilt(self._config.TURRET_START_TILT)
        time.sleep(1)

    def _step(self):
        self._turret.set_tilt(self._turret.current_tilt + self._tilt_dir)
        self._turret.set_pan(self._turret.current_pan + self._pan_dir)

        if self._turret.current_pan in (
            self._config.TURRET_MIN_PAN,
            self._config.TURRET_MAX_PAN,
        ):
            self._pan_dir *= -1
        if self._turret.current_tilt in (
            self._config.TURRET_MIN_TILT,
            self._config.TURRET_MAX_TILT,
        ):
            self._tilt_dir *= -1

        time.sleep(0.01)
        return True
