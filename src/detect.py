import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import os

# --- Configuration ---
MODEL_PATH = "models/detect.tflite"      # Path to your TFLite model
LABELS_PATH = "models/labelmap.txt"      # Path to label file
TEST_IMAGES_DIR = "data/test_images/"    # Folder with images to test
OUTPUT_DIR = "data/output/"              # Folder to save annotated images
CONFIDENCE_THRESHOLD = 0.5               # Minimum score to consider a detection

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load labels ---
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()

def detect_movement(image1, image2, threshold=30):
    """Detects movement between two images by comparing pixel differences."""
    # Convert images to grayscale
    img1_gray = image1.convert("L")
    img2_gray = image2.convert("L")

    # Compute absolute difference between images
    diff = np.abs(np.array(img1_gray, dtype=np.int16) - np.array(img2_gray, dtype=np.int16))
    
    # Threshold the difference to get binary image
    movement_mask = (diff > threshold).astype(np.uint8) * 255
    
    # Calculate the percentage of changed pixels
    movement_percentage = np.sum(movement_mask) / (movement_mask.shape[0] * movement_mask.shape[1] * 255)
    
    return movement_percentage > 0.01  # Return True if more than 1% of pixels changed

# --- Detection function ---
def detect_objects(image_path):
    # Load image
    img = Image.open(image_path).convert("RGB")
    input_shape = input_details['shape']
    resized_img = img.resize((input_shape[2], input_shape[1]))
    input_data = np.expand_dims(np.array(resized_img, dtype=np.uint8), axis=0)

    # Run inference
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # Extract outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]   # bounding boxes
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # confidence scores

    # Draw results
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i in range(len(scores)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            width, height = img.size
            (left, right, top, bottom) = (xmin*width, xmax*width, ymin*height, ymax*height)
            draw.rectangle([left, top, right, bottom], outline="red", width=2)
            label = f"{labels[int(class_ids[i])]}: {scores[i]:.2f}"
            draw.text((left, top-10), label, fill="red", font=font)
            print(f"Detected {label} in {image_path}")

    # Save annotated image
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    img.save(output_path)

# --- Run detection on all test images ---
for filename in os.listdir(TEST_IMAGES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        detect_objects(os.path.join(TEST_IMAGES_DIR, filename))
