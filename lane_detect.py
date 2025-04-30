import cv2
import numpy as np

# Load classifiers
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
stop_cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml.xml')  # Make sure the path and filename are correct

# Video source
cap = cv2.VideoCapture("test_video2.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (width // 2 - 70, height // 2 + 50),
        (width // 2 + 70, height // 2 + 50),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, np.array([]), minLineLength=60, maxLineGap=150)
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))

        for x1, y1, x2, y2 in left_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        for x1, y1, x2, y2 in right_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    direction = "Keep Straight"
    if len(left_lines) > 0 and len(right_lines) == 0:
        direction = "Turn Right"
    elif len(right_lines) > 0 and len(left_lines) == 0:
        direction = "Turn Left"

    return frame, direction

def detect_pedestrians(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, "Pedestrian", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame, len(pedestrians)

def detect_stop_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stop_signs = stop_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in stop_signs:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(frame, "STOP SIGN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame, len(stop_signs)

def detect_traffic_lights(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([45, 100, 100]), np.array([70, 255, 255]))

    red = cv2.countNonZero(red_mask) > 500
    yellow = cv2.countNonZero(yellow_mask) > 500
    green = cv2.countNonZero(green_mask) > 500

    if red:
        return frame, "STOP"
    elif yellow:
        return frame, "SLOW DOWN"
    elif green:
        return frame, "GO"
    else:
        return frame, "UNKNOWN"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    frame, direction = detect_lanes(frame)
    frame, pedestrian_count = detect_pedestrians(frame)
    frame, stop_signs = detect_stop_signs(frame)
    frame, traffic_light_status = detect_traffic_lights(frame)

    # HUD Overlay
    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Pedestrians: {pedestrian_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Traffic Light: {traffic_light_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    if stop_signs > 0:
        cv2.putText(frame, "ALERT: STOP SIGN AHEAD", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if 'out' not in locals():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_simulation.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    out.write(frame)


out.release()
cap.release()
cv2.destroyAllWindows()
