import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
base_dir = os.path.dirname(os.path.abspath(__file__))
sunglasses = cv2.imread("sunglass1.jpeg", cv2.IMREAD_UNCHANGED)
beard = cv2.imread("beard.png", cv2.IMREAD_UNCHANGED)


def apply_overlay(frame, overlay, x, y, w, h):


    resized = cv2.resize(overlay, (w, h))

    if resized.ndim == 2:
        overlay_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        mask = np.full((h, w), 255, dtype=np.uint8)
    else:
        channels = resized.shape[2]
        if channels == 4:
            overlay_img = resized[:, :, :3]
            mask = resized[:, :, 3]
        elif channels == 3:
            overlay_img = resized
            mask = np.full((h, w), 255, dtype=np.uint8)
        elif channels == 2:
            overlay_img = cv2.cvtColor(resized[:, :, 0], cv2.COLOR_GRAY2BGR)
            mask = resized[:, :, 1]
        elif channels == 1:
            overlay_img = cv2.cvtColor(resized[:, :, 0], cv2.COLOR_GRAY2BGR)
            mask = np.full((h, w), 255, dtype=np.uint8)
        else:
            return

    mask_inv = cv2.bitwise_not(mask)
    roi = frame[y:y + h, x:x + w]
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)
    frame[y:y + h, x:x + w] = cv2.add(bg, fg)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    flag, frame = cap.read()
    if not flag:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        sunglass_x = x
        sunglass_y = y + h // 4
        sunglass_w = w
        sunglass_h = int(h * 0.4)
        apply_overlay(frame, sunglasses, sunglass_x, sunglass_y, sunglass_w, sunglass_h)

        beard_x = x + int(w * 0.1)
        beard_y = y + int(h * 0.6)
        beard_w = int(w * 0.8)
        beard_h = int(h * 0.35)
        apply_overlay(frame, beard, beard_x, beard_y, beard_w, beard_h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("sunglasses filter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()