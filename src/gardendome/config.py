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
TURRET_DEAD_ZONE = 20
TURRET_SETTLE_FRAMES = 3
TURRET_AIM_ADJUST = 5
