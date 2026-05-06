import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("weather_model.h5")
print("model loaded successfully...")

class_names = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

video_source = "https://www.youtube.com/watch?v=Ln7zxnl4Pbc"

def open_video_source(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source)

    if isinstance(source, str) and "youtube.com" in source:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise RuntimeError(
                "YouTube URL detected, but yt-dlp is not installed. "
                "Install it with: pip install yt-dlp"
            ) from exc

        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=False)
            stream_url = info["url"]
        return cv2.VideoCapture(stream_url)

    return cv2.VideoCapture(source)

cap = open_video_source(video_source)

if not cap.isOpened():
    raise RuntimeError("Unable to open video stream.")

while True:
    flag, frame = cap.read()
    if not flag:
        break

    img = cv2.resize(frame, (128, 128))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_label = class_names[predicted_index]

    confidence_score = np.max(predictions[0])

    cv2.putText(
        frame,
        f"{predicted_label}: {confidence_score:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Weather detection : ", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()