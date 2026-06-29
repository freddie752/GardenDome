from gardendome.turret_control.PCA9685 import PCA9685


from dataclasses import dataclass


@dataclass
class TurretConfig:
    min_tilt: int
    max_tilt: int
    min_pan: int
    max_pan: int
    start_tilt: int
    start_pan: int
    dead_zone: int
    settle_frames: int
    aim_adjust: int


class Turret:
    def __init__(self, logger, turret_config):
        self._logger = logger
        self._pwm = PCA9685()
        self._pwm.setPWMFreq(50)
        self._min_tilt = turret_config.min_tilt
        self._max_tilt = turret_config.max_tilt
        self._min_pan = turret_config.min_pan
        self._max_pan = turret_config.max_pan
        self._dead_zone = turret_config.dead_zone
        self._settle_frames = turret_config.settle_frames
        self._aim_adjust = turret_config.aim_adjust
        self._moved = False
        self._current_tilt = None
        self._current_pan = None
        self.set_tilt(turret_config.start_tilt)
        self.set_pan(turret_config.start_pan)

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

    @property
    def moved(self):
        return self._moved

    def aim(self, centre, frame_shape):
        centre_x, centre_y = centre
        frame_h, frame_w = frame_shape[:2]
        frame_centre_x = frame_w / 2
        frame_centre_y = frame_h / 2
        self._moved = False

        if centre_x < frame_centre_x - self._dead_zone:
            self.set_pan(self.current_pan - self._aim_adjust)
            self._moved = True
        elif centre_x > frame_centre_x + self._dead_zone:
            self.set_pan(self.current_pan + self._aim_adjust)
            self._moved = True

        if centre_y < frame_centre_y - self._dead_zone:
            self.set_tilt(self.current_tilt - self._aim_adjust)
            self._moved = True
        elif centre_y > frame_centre_y + self._dead_zone:
            self.set_tilt(self.current_tilt + self._aim_adjust)
            self._moved = True
