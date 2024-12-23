from flask import Flask, jsonify, request

import cv2
import mediapipe
import pyautogui

from enum import Enum
from mediapipe.python.solutions.hands import HandLandmark

app = Flask(__name__)


@app.route('/landmarks', methods=['GET'])
def get_landmarks():
    hand_landmark_dict = {key: value.value for key, value in HandLandmark.__members__.items()}
    return jsonify(hand_landmark_dict)


@app.route('/event', methods=['GET'])
def get_events():
    hand_landmark_dict = {key: value.value for key, value in HandLandmark.__members__.items()}
    return jsonify(hand_landmark_dict)


class Event:
    point_1: str
    point_2: str
    event: str


class ClickState(Enum):
    DISABLED = 0  # 클릭 비활성
    READY = 1  # 클릭 준비 완료
    BUSY = 2  # 클릭 작동 중


def __main__():
    mp_hands = mediapipe.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mediapipe.solutions.drawing_utils

    screen_w, screen_h = pyautogui.size()

    read_click: ClickState = ClickState.READY

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame_h, frame_w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                gum_gi = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                gum_gi_x, gum_gi_y = int(gum_gi.x * screen_w), int(gum_gi.y * screen_h)

                if handedness.classification[0].label == 'Right':
                    pyautogui.moveTo(gum_gi_x, gum_gi_y)

                    if read_click:
                        thumb_tib = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

                        thumb_tib_X = int(thumb_tib.x * screen_w)
                        thumb_ip_X = int(thumb_ip.x * screen_w)

                        if thumb_ip_X < thumb_tib_X:
                            cv2.circle(frame, (0, 0), 100, (0, 255, 0), -1)
                            pyautogui.rightClick()

                elif handedness.classification[0].label == 'Left':
                    pass

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) != -1:
            break


if __name__ == '__main__':
    __main__()
    # app.run()
