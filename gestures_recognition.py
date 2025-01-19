import cv2
import cv2.plot
import numpy as np
import time
import math
import mediapipe as mp
import matplotlib.pyplot as plt
import websockets
import asyncio
import json
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
import sys
import signal


connected_clients = set()
data = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
}
# Flaga do kontrolowania zakończenia wątków

terminate_flag = False

def signal_handler(sig, frame):
    global terminate_flag
    print("\nCtrl+C detected! Shutting down...")
    terminate_flag = True
    sys.exit(0)

# Przechwytywanie sygnału Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

async def video_stream(websocket):
    # Dodanie klienta do listy podłączonych
    print(f"Nowy klient podłączony: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        while True:
            # Sprawdzenie połączenia co pewien czas
            print(f'Wysyłanie danych: {data}')
            await websocket.send(json.dumps(data))
            await asyncio.sleep(1)
    except websockets.exceptions.ConnectionClosed:
        print(f"Klient rozłączony: {websocket.remote_address}")
    finally:
        connected_clients.remove(websocket)

class Kalman_Filtering:

    def __init__(self, n_points):
        self.n_points = n_points

    def initialize(self):
        dt = 1
        n_states = self.n_points * 4
        n_measures = self.n_points * 2
        self.kalman = cv2.KalmanFilter(n_states, n_measures)
        kalman = self.kalman
        kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
        kalman.measurementNoiseCov = np.eye(n_measures, dtype=np.float32) * 0.0005
        kalman.measurementMatrix = np.zeros((n_measures, n_states), np.float32)

        self.Measurement_array = []
        self.dt_array = []

        for i in range(0, n_states, 4):
            self.Measurement_array.append(i)
            self.Measurement_array.append(i + 1)

        for i in range(0, n_states):
            if i not in self.Measurement_array:
                self.dt_array.append(i)

        for i, j in zip(self.Measurement_array, self.dt_array):
            kalman.transitionMatrix[i, j] = dt;

        for i in range(0, n_measures):
            kalman.measurementMatrix[i, self.Measurement_array[i]] = 1

    def predict(self, points, dt):
        for i, j in zip(self.Measurement_array, self.dt_array):
            self.kalman.transitionMatrix[i, j] = dt;

        pred = []
        input_points = np.float32(np.ndarray.flatten(points))
        self.kalman.correct(input_points)
        tp = self.kalman.predict()

        for i in self.Measurement_array:
            pred.append(int(tp[i]))

        return pred
#####Koniec Klasy Kalman_Filtering

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def send_velocity_data():
    print('Invoking socket handler')
    cap = cv2.VideoCapture(0)
    print('Camera initialized')

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    kalmans = []
    for i in range(21):
        kalmans.append(Kalman_Filtering(1))
        kalmans[i].initialize()

    prev_time = time.time()

    canvas = None
    drawing = False

    print('Before get image')
    dron = cv2.imread('dron.png', cv2.IMREAD_UNCHANGED)

    if dron is None:
        print("Błąd wczytywania obrazu!")
    else:
        dron = cv2.resize(dron, (100, 100))
        if dron.shape[2] == 4:  
            dron_rgb = dron  
        else:
            dron_rgb = cv2.cvtColor(dron, cv2.COLOR_BGR2BGRA)

    # Początkowa pozycja drona
    dron_x, dron_y = 260, 380
    last_dron_x, last_dron_y = dron_x, dron_y

    while not terminate_flag:
        success, img = cap.read()

        if canvas is None:
            canvas = np.zeros_like(img)

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time        

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        h, w, c = img.shape
        center_x, center_y = w // 2, h // 2

        dron_h, dron_w, _ = dron_rgb.shape
        y_offset = (h - dron_h)
        x_offset = (w - dron_w) // 2

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]  
            h, w, c = img.shape

            thumb_tip = (int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h))
            index_tip = (int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h))
            cx, cy = thumb_tip

            if drawing == False and distance(thumb_tip, index_tip) < 50:
                cords = np.array([[cx, cy]], dtype=np.int64)

            if distance(thumb_tip, index_tip) < 50:
                drawing = True
                x, y = cords[0]
                cv2.circle(img, (x,y), 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (x, y), (cx, cy), (0, 255, 0), 3)

                # Użyj punktów przewidywanych przez filtr Kalmana
                points = np.array([[cx, cy]], dtype=np.float32)
                kx, ky = kalmans[8].predict(points, dt)  

                velocity = np.array([[(kx-x)*0.1,(ky-y)*0.1]], dtype=np.int64)  

                dron_x += int(velocity[0][0])
                dron_y += int(velocity[0][1])

                dron_x = np.clip(dron_x, 0, w - dron_w)  
                dron_y = np.clip(dron_y, 0, h - dron_h)  

                last_dron_x, last_dron_y = dron_x, dron_y

                #TO JEST NA WYJSCIE
                # print(f"Velocity: {velocity[0][0]}, {velocity[0][1]}")

                # Wysyłanie danych prędkości przez WebSocket
                velocity_data = {
                    "y": -1 * (float(velocity[0][0]) / 100),
                    "z": -1 * (float(velocity[0][1]) / 100),
                    "x": 0.0
                }
                global data
                data = velocity_data
                # if connected_clients:
                #     message = json.dumps(velocity_data)
                #     await asyncio.gather(
                #         *[client.send(message) for client in connected_clients],
                #         return_exceptions=True
                #     )


            else:
                drawing = False
                dron_x, dron_y = last_dron_x, last_dron_y

            cv2.putText(img, str(drawing), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                points = np.array([[cx, cy]], dtype=np.float32)

                color2 = (0, 255, 0)  
                kx, ky = kalmans[id].predict(points, dt)
                cv2.circle(img, (kx, ky), 3, color2, cv2.FILLED)
                    
        dron_rgb_resized = cv2.resize(dron_rgb, (dron_w, dron_h))  

        for c in range(0, 3):  
            img[dron_y:dron_y + dron_h, dron_x:dron_x + dron_w, c] = \
                img[dron_y:dron_y + dron_h, dron_x:dron_x + dron_w, c] * (1 - dron_rgb_resized[:, :, 3] / 255.0) + \
                dron_rgb_resized[:, :, c] * (dron_rgb_resized[:, :, 3] / 255.0)

        img = cv2.addWeighted(img, 1, canvas, 0.5, 0)

        img = cv2.flip(img, 1)

        cv2.imshow("Drawing", img)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

HOST = '127.0.0.1'
PORT = 8080

async def main():
    threading.Thread(target=send_velocity_data).start()
    async with websockets.serve(video_stream, HOST, PORT):
        await asyncio.get_running_loop().create_future()
    # threading.Thread(target=server.run_forever, args=(HOST, PORT)).start()
    # Równoczesne uruchamianie serwera i przesyłania obrazu

if __name__ == "__main__":
    asyncio.run(main())