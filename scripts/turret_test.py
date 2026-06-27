import time
from turret_control.PCA9685 import PCA9685

MAX_TILT = 175
MIN_TILT = 65
MAX_PAN = 180
MIN_PAN = 0
pwm = PCA9685()
pwm.setPWMFreq(50)
pan_dir = 1
tilt_dir = 1
pan_loc = MIN_PAN
tilt_loc = MIN_TILT


while True:
    pwm.setRotationAngle(0, tilt_loc)
    pwm.setRotationAngle(1, pan_loc)
    pan_loc += pan_dir
    tilt_loc += tilt_dir
    if(pan_loc == MAX_PAN) or (pan_loc == MIN_PAN):
        pan_dir = pan_dir * -1
    if(tilt_loc == MAX_TILT) or (tilt_loc == MIN_TILT):
        tilt_dir = tilt_dir * -1
    time.sleep(0.01)
pwm.setRotationAngle(1, 90)
time.sleep(1)

# for i in range(0,180,10):
#     pwm.setRotationAngle(1, i)
#     time.sleep(0.5)
#     
#     
# for i in range(180,0,-10):
#     pwm.setRotationAngle(1, i) 
#     time.sleep(0.5)
#      



# try:
#     print ("Start turret orientation.")    
#     pwm.setPWMFreq(50)
#     #pwm.setServoPulse(1,500) 
#     pwm.setRotationAngle(1, 0)
#     
#     while True:
#         # setServoPulse(2,2500)
#         for i in range(10,170,1): 
#             pwm.setRotationAngle(1, i)   
#             if(i<80):
#                 pwm.setRotationAngle(0, i)   
#             time.sleep(0.1)
# 
#         for i in range(170,10,-1): 
#             pwm.setRotationAngle(1, i)   
#             if(i<80):
#                 pwm.setRotationAngle(0, i)            
#             time.sleep(0.1)
# 
# except:
#     pwm.exit_PCA9685()
#     print("\nProgram end")
#     exit()