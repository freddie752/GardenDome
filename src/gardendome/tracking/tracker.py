import cv2


class Tracker:
    @staticmethod
    def calculate_centre(bbox):
        """Returns the (x, y) centre of a pixel (x, y, w, h) box."""
        x, y, w, h = bbox
        return x + w // 2, y + h // 2

    @staticmethod
    def draw_box(frame, bbox):
        """Draws a pixel (x, y, w, h) box on the frame."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 5)
        return frame

    @staticmethod
    def draw_centre(frame, centre):
        """Draws a dot at the (x, y) centre point on the frame."""
        cx, cy = int(centre[0]), int(centre[1])
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        return frame

    @staticmethod
    def draw_label(frame, bbox, text):
        """Draws text above the top-left corner of a pixel (x, y, w, h) box."""
        x, y, _, _ = bbox
        cv2.putText(
            frame, text, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        return frame

    @staticmethod
    def draw_detection(frame, detection):
        frame = Tracker.draw_box(frame, detection.bbox)
        label = f"{detection.label} {detection.score:.0%}"
        return Tracker.draw_label(frame, detection.bbox, label)

