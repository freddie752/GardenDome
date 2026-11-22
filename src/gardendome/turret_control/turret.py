from dataclasses import dataclass

from gardendome.turret_control.angles import Angles
from gardendome.vendor.PCA9685 import PCA9685


@dataclass
class TurretConfig:
    min_pan: int
    max_pan: int
    min_tilt: int
    max_tilt: int
    start_pan: int
    start_tilt: int


class Turret:
    def __init__(self, logger, turret_config):
        self._logger = logger
        self._pwm = PCA9685()
        self._pwm.setPWMFreq(50)
        self._min_pan = turret_config.min_pan
        self._max_pan = turret_config.max_pan
        self._min_tilt = turret_config.min_tilt
        self._max_tilt = turret_config.max_tilt
        self._current_pan = None
        self._current_tilt = None
        self.set_pan(turret_config.start_pan)
        self.set_tilt(turret_config.start_tilt)

    @property
    def current_pan(self):
        return self._current_pan

    @property
    def current_tilt(self):
        return self._current_tilt

    @property
    def position(self):
        return Angles(pan=self._current_pan, tilt=self._current_tilt)

    def set_pan(self, pan_angle):
        if self._min_pan <= pan_angle <= self._max_pan:
            self._pwm.setRotationAngle(1, pan_angle)
            self._current_pan = pan_angle
        else:
            self._logger.info("Invalid pan coord")

    def set_tilt(self, tilt_angle):
        if self._min_tilt <= tilt_angle <= self._max_tilt:
            self._pwm.setRotationAngle(0, tilt_angle)
            self._current_tilt = tilt_angle
        else:
            self._logger.info("Invalid tilt coord")

    def move_to(self, position):
        pan_angle = max(min(position.pan, self._max_pan), self._min_pan)
        tilt_angle = max(min(position.tilt, self._max_tilt), self._min_tilt)
        if pan_angle != self.current_pan:
            self.set_pan(pan_angle)
        if tilt_angle != self.current_tilt:
            self.set_tilt(tilt_angle)
