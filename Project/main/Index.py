import time

from flask import Flask, Response, render_template
import cv2
import tensorflow as tf
import numpy as np
from statistics import mode

app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
text = ""
model = tf.keras.models.load_model("model")
i = 0
list = []


alpha = {
    0 : "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "f",
    6: "g",
    7: "h",
    8: "i",
    9: "j",
    10: "k",
    11: "l",
    12: "m",
    13: "n",
    14: "o",
    15: "p",
    16: "q",
    17: "r",
    18: "s",
    19: "t",
    20: "u",
    21: "v",
    22: "w",
    23: "x",
    24: "y",
    25: "z",
    26: "",
    27: " "
}

@app.route('/')
def index():
    return render_template('index.html', txt = text)


def generate_frame():
    global text
    global i
    while True:
        success, frame=camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)


        inp = cv2.resize(frame, (112, 112))
        rgb_tensor = tf.convert_to_tensor(inp, dtype=tf.float32)
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
        ynew = model.predict(rgb_tensor/255.)
        y_classes = np.argmax(ynew)
        list.append(alpha[y_classes])
        if i >= 100:
            text += mode(list)
            list.clear()
            i = 0

        rgb_tensor = buffer.tobytes()
        i+=1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + rgb_tensor + b'\r\n\r\n')

def generate_text():
    yield text


@app.route('/video')
def video():
    global camera
    return Response(generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()