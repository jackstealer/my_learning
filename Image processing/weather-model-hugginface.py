import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# =========================
# LOAD TRAINED MODEL
# =========================
model = load_model("weather_classifier.keras")

# =========================
# CLASS NAMES
# IMPORTANT:
# Keep same order as training
# =========================
class_names = [
    'dew',
    'fogsmog',
    'frost',
    'glaze',
    'hail',
    'lightning',
    'rain',
    'rainbow',
    'rime',
    'sandstorm',
    'snow'
]

# =========================
# VIDEO PATH
# =========================
video_path = "weather-video.mp4"

# =========================
# OPEN VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# =========================
# OUTPUT VIDEO SETTINGS
# =========================
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    fps,
    (frame_width, frame_height)
)

# =========================
# PROCESS VIDEO
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for model
    resized = cv2.resize(frame, (224, 224))

    # Convert to numpy array
    img_array = np.array(resized, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess for ResNet
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)

    # =========================
    # DRAW RESULT ON FRAME
    # =========================
    text = f"{predicted_class} ({confidence:.2f})"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =========================
    # SHOW VIDEO
    # =========================
    cv2.imshow("Weather Prediction", frame)

    # =========================
    # SAVE OUTPUT VIDEO
    # =========================
    out.write(frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE RESOURCES
# =========================
cap.release()
out.release()

cv2.destroyAllWindows()

print("Processing Complete")
print("Saved as output.mp4")