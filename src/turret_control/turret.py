import time
from PCA9685 import PCA9685

MAX_TILT = 175
MIN_TILT = 65
MAX_PAN = 180
MIN_PAN = 0

START_PAN = 90
START_TILT = 120

class Turret():
    
    def __init__(self):
        self.pwm = PCA9685()
        self.pwm.setPWMFreq(50)
        self.pwm.setRotationAngle(0, START_TILT)
        self.pwm.setRotationAngle(1, START_PAN)
        
    def tilt(self, tilt_coord):
        if (MIN_TILT <= tilt_coord <= MAX_TILT):
            self.pwm.setRotationAngle(0, tilt_coord)
        else:
            print("Invalid tilt coord")
            
    def pan(self, pan_coord):
        if (MIN_PAN <= pan_coord <= MAX_PAN):
            self.pwm.setRotationAngle(1, pan_coord)
        else:
            print("Invalid pan coord")
        

t = Turret()