import cv2

class Tracker:
    def draw_box(self, frame, bbox):
        x, y, w, h = bbox
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 5)
        return annotated

    def calculate_centre(self, bbox):
        x, y, w, h = bbox
        return x + 0.5 * w, y + 0.5 * h
