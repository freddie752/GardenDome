from gardendome.turret_control.PCA9685 import PCA9685


class Turret:
    def __init__(
        self, logger, min_tilt, max_tilt, min_pan, max_pan, start_tilt, start_pan
    ):
        self._logger = logger
        self._pwm = PCA9685()
        self._pwm.setPWMFreq(50)
        self._min_tilt = min_tilt
        self._max_tilt = max_tilt
        self._min_pan = min_pan
        self._max_pan = max_pan
        self._current_tilt = None
        self._current_pan = None
        self.set_tilt(start_tilt)
        self.set_pan(start_pan)

    def set_tilt(self, tilt_coord):
        if self._min_tilt <= tilt_coord <= self._max_tilt:
            self._pwm.setRotationAngle(0, tilt_coord)
            self._current_tilt = tilt_coord
        else:
            self._logger.info("Invalid tilt coord")

    @property
    def current_tilt(self):
        return self._current_tilt

    def set_pan(self, pan_coord):
        if self._min_pan <= pan_coord <= self._max_pan:
            self._pwm.setRotationAngle(1, pan_coord)
            self._current_pan = pan_coord
        else:
            self._logger.info("Invalid pan coord")

    @property
    def current_pan(self):
        return self._current_pan
