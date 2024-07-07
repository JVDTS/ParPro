from flask import Flask, Response, jsonify, render_template
import cv2
import pickle
import cvzone
import numpy as np
from pyngrok import ngrok
import threading
import time
import json

app = Flask(__name__)

# Load the car park position
with open('Carpark', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

cap = cv2.VideoCapture('carPark.mp4')


free_spaces = 0

# JSON file to store the free spaces count
json_file = 'free_spaces.json'


def checkParkingSpace(imgPro, img):
    global free_spaces
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (128, 0, 128)
            thickness = 1
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 1

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=1, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=4, offset=20, colorR=(255, 200, 0))

    free_spaces = spaceCounter
    return spaceCounter


def generate_frames():
    global free_spaces
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            break

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        free_spaces = checkParkingSpace(imgDilate, img)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def update_free_spaces():
    global free_spaces
    while True:
        # Update free spaces count
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            break

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        free_spaces = checkParkingSpace(imgDilate, img)

        # Write to JSON file
        with open(json_file, 'w') as file:
            json.dump({'free_spaces': free_spaces}, file)

        time.sleep(5)  # Wait for 5 seconds before updating again


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/free_spaces_count')
def free_spaces_count():
    with open(json_file, 'r') as file:
        data = json.load(file)
    return jsonify(data)


if __name__ == '__main__':
    try:
        # My  ngrok  an auth token
        ngrok.set_auth_token('2iTVVqaES8LU2TcteQnDKRxI9if_6jeMYJg1k8sSKnPLUAUru')

        # This helps to expose Flask app on port 5000
        public_url = ngrok.connect(5000).public_url
        print(" * ngrok tunnel available at:", public_url)

       # background thread
        threading.Thread(target=update_free_spaces, daemon=True).start()

        # begins Flask app locally
        app.run(port=5000)

    except Exception as e:
        print(" * Error starting Ngrok tunnel:", e)
