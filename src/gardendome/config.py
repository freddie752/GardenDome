# Recording
RECORDING_DIR = "data/recording/"
BITRATE = 2000000

# Motion detection
MOTION_THRESHOLD = 50  # Pixel intensity difference to detect
MOTION_FRACTION = 0.01  # Fraction of pixels change to detect

# Object detection
DETECTOR_MODEL_PATH = "models/ssd_mobilenet_v1/detect.tflite"
DETECTOR_LABELS_PATH = "models/ssd_mobilenet_v1/labelmap.txt"
DETECTOR_SCORE_THRESHOLD = 0.5
DETECTOR_NUM_THREADS = 4
DETECTOR_TARGET_LABEL = "person"

# Slack
SLACK_LOGGING = True
SLACK_CHANNEL = "C09M5FPTSLV"

# Turret
TURRET_MAX_TILT = 175
TURRET_MIN_TILT = 65
TURRET_MAX_PAN = 180
TURRET_MIN_PAN = 0
TURRET_START_PAN = 90
TURRET_START_TILT = 120
TURRET_AIM_ADJUST = 1

# Camera
CAMERA_FOV_H = 47  # Horizontal field of view in degrees (measured on the Pi)
CAMERA_FOV_V = 29  # Vertical field of view in degrees (measured on the Pi)

# Following
FOLLOW_GAIN = 0.5  # Fraction of the error closed per step
FOLLOW_MAX_STEP = 8  # Largest step in degrees
FOLLOW_START_THRESHOLD = 3  # Error in degrees that starts a move when idle
FOLLOW_STOP_THRESHOLD = 1  # Error in degrees that stops a move
FOLLOW_SETTLE_TIME = 0.15  # Seconds to ignore detections after a move
