import sys
import tty
import termios
from gardendome import config
from gardendome.turret_control.turret import Turret, TurretConfig


def main():
    turret_config = TurretConfig(
        min_tilt=config.TURRET_MIN_TILT,
        max_tilt=config.TURRET_MAX_TILT,
        min_pan=config.TURRET_MIN_PAN,
        max_pan=config.TURRET_MAX_PAN,
        start_tilt=config.TURRET_START_TILT,
        start_pan=config.TURRET_START_PAN,
        dead_zone=config.TURRET_DEAD_ZONE,
        aim_adjust=config.TURRET_AIM_ADJUST,
        settle_frames=config.TURRET_SETTLE_FRAMES,
    )
    logger = type("Logger", (), {"info": print})()
    turret = Turret(logger=logger, turret_config=turret_config)

    print("Turret manual control")
    print("w=tilt up, s=tilt down, a=pan left, d=pan right, q=quit")
    print(f"Current: pan={turret.current_pan}, tilt={turret.current_tilt}")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1)
            if key == "w":
                turret.set_tilt(turret.current_tilt - config.TURRET_AIM_ADJUST)
            elif key == "s":
                turret.set_tilt(turret.current_tilt + config.TURRET_AIM_ADJUST)
            elif key == "a":
                turret.set_pan(turret.current_pan + config.TURRET_AIM_ADJUST)
            elif key == "d":
                turret.set_pan(turret.current_pan - config.TURRET_AIM_ADJUST)
            elif key == "q":
                break
            print(
                f"\rCurrent: pan={turret.current_pan}, tilt={turret.current_tilt}   ",
                end="",
            )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
