from unittest.mock import MagicMock, patch

from gardendome.turret_control.angles import Angles
from gardendome.turret_control.turret import Turret, TurretConfig

VALID_CONFIG = TurretConfig(
    min_tilt=65,
    max_tilt=175,
    min_pan=0,
    max_pan=180,
    start_tilt=120,
    start_pan=90,
)


def make_turret():
    logger = MagicMock()
    mock_pwm = MagicMock()
    with patch("gardendome.turret_control.turret.PCA9685", return_value=mock_pwm):
        turret = Turret(logger=logger, turret_config=VALID_CONFIG)
    turret._pwm = MagicMock()
    turret._logger = logger
    return turret


def make_turret_with_pwm():
    logger = MagicMock()
    mock_pwm = MagicMock()
    with patch("gardendome.turret_control.turret.PCA9685", return_value=mock_pwm):
        turret = Turret(logger=logger, turret_config=VALID_CONFIG)
    turret._logger = logger
    return turret, mock_pwm


def test_init_sets_current_tilt_to_start():
    turret, _ = make_turret_with_pwm()
    assert turret.current_tilt == VALID_CONFIG.start_tilt


def test_init_sets_current_pan_to_start():
    turret, _ = make_turret_with_pwm()
    assert turret.current_pan == VALID_CONFIG.start_pan


def test_init_sets_pwm_frequency():
    turret, mock_pwm = make_turret_with_pwm()
    mock_pwm.setPWMFreq.assert_called_once_with(50)


def test_init_sends_start_position_to_pwm():
    turret, mock_pwm = make_turret_with_pwm()
    mock_pwm.setRotationAngle.assert_any_call(0, VALID_CONFIG.start_tilt)
    mock_pwm.setRotationAngle.assert_any_call(1, VALID_CONFIG.start_pan)


def test_set_tilt_sends_angle_to_pwm():
    turret = make_turret()
    turret.set_tilt(90)
    turret._pwm.setRotationAngle.assert_called_once_with(0, 90)


def test_set_tilt_updates_current_tilt():
    turret = make_turret()
    turret.set_tilt(90)
    assert turret.current_tilt == 90


def test_set_tilt_logs_when_out_of_bounds():
    turret = make_turret()
    turret.set_tilt(0)
    turret._logger.info.assert_called_once()


def test_set_tilt_does_not_update_current_tilt_when_out_of_bounds():
    turret = make_turret()
    original_tilt = turret.current_tilt
    turret.set_tilt(0)
    assert turret.current_tilt == original_tilt


def test_set_tilt_accepts_min_boundary():
    turret = make_turret()
    turret.set_tilt(VALID_CONFIG.min_tilt)
    assert turret.current_tilt == VALID_CONFIG.min_tilt


def test_set_tilt_accepts_max_boundary():
    turret = make_turret()
    turret.set_tilt(VALID_CONFIG.max_tilt)
    assert turret.current_tilt == VALID_CONFIG.max_tilt


def test_set_pan_sends_angle_to_pwm():
    turret = make_turret()
    turret.set_pan(45)
    turret._pwm.setRotationAngle.assert_called_once_with(1, 45)


def test_set_pan_updates_current_pan():
    turret = make_turret()
    turret.set_pan(45)
    assert turret.current_pan == 45


def test_set_pan_logs_when_out_of_bounds():
    turret = make_turret()
    turret.set_pan(270)
    turret._logger.info.assert_called_once()


def test_set_pan_does_not_update_current_pan_when_out_of_bounds():
    turret = make_turret()
    original_pan = turret.current_pan
    turret.set_pan(270)
    assert turret.current_pan == original_pan


def test_set_pan_accepts_min_boundary():
    turret = make_turret()
    turret.set_pan(VALID_CONFIG.min_pan)
    assert turret.current_pan == VALID_CONFIG.min_pan


def test_set_pan_accepts_max_boundary():
    turret = make_turret()
    turret.set_pan(VALID_CONFIG.max_pan)
    assert turret.current_pan == VALID_CONFIG.max_pan


def test_position_returns_current_angles():
    turret = make_turret()
    turret.move_to(Angles(pan=45, tilt=100))
    assert turret.position == Angles(pan=45, tilt=100)


def test_move_to_updates_current_angles():
    turret = make_turret()
    turret.move_to(Angles(pan=45, tilt=100))
    assert turret.current_pan == 45
    assert turret.current_tilt == 100


def test_move_to_clamps_above_max():
    turret = make_turret()
    turret.move_to(Angles(pan=200, tilt=200))
    assert turret.current_pan == VALID_CONFIG.max_pan
    assert turret.current_tilt == VALID_CONFIG.max_tilt


def test_move_to_clamps_below_min():
    turret = make_turret()
    turret.move_to(Angles(pan=-20, tilt=0))
    assert turret.current_pan == VALID_CONFIG.min_pan
    assert turret.current_tilt == VALID_CONFIG.min_tilt


def test_move_to_does_not_write_when_already_there():
    turret = make_turret()
    turret.move_to(Angles(pan=VALID_CONFIG.start_pan, tilt=VALID_CONFIG.start_tilt))
    turret._pwm.setRotationAngle.assert_not_called()


def test_move_to_does_not_write_when_at_limit_and_target_beyond():
    turret = make_turret()
    turret.move_to(Angles(pan=VALID_CONFIG.max_pan, tilt=VALID_CONFIG.max_tilt))
    turret._pwm.setRotationAngle.reset_mock()
    turret.move_to(Angles(pan=200, tilt=200))
    turret._pwm.setRotationAngle.assert_not_called()


def test_move_to_only_writes_pan_when_tilt_unchanged():
    turret = make_turret()
    turret.move_to(Angles(pan=45, tilt=VALID_CONFIG.start_tilt))
    turret._pwm.setRotationAngle.assert_called_once_with(1, 45)


def test_move_to_only_writes_tilt_when_pan_unchanged():
    turret = make_turret()
    turret.move_to(Angles(pan=VALID_CONFIG.start_pan, tilt=100))
    turret._pwm.setRotationAngle.assert_called_once_with(0, 100)
