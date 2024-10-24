from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from flask import Flask, render_template, Response
import cv2
import numpy as np
import mss

app = Flask(__name__)

# Load the face classifier and the emotion detection model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

# Emotion labels
class_emotion = ['Bored', 'Happy', 'Neutral', 'Confused', 'Surprise']

# Function to capture screen frames
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Adjust this if you have multiple monitors (1 for primary, 2 for secondary, etc.)
        while True:
            screen = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR for OpenCV
            yield frame

# Function to generate frames with emotion detection
def generate_frames():
    for frame in capture_screen():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 7)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                emotion = class_emotion[preds.argmax()]  # Get the predicted emotion
                emotion_position = (x, y)
                cv2.putText(frame, emotion, emotion_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Encode the frame and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('emotion.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)