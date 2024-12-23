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
                # 엄지
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                # 검지
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # 중지
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                # 약지
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                # 소지
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # 오른손(마우스)
                if handedness.classification[0].label == 'Right':

                    # 마우스 이동을 위한 검지 좌표 정규화 값
                    index_finger_tip_x = int(index_finger_tip.x * screen_w)
                    index_finger_tip_y = int(index_finger_tip.y * screen_h)

                    # 마우스 이동 이벤트(상시 작동)
                    pyautogui.moveTo(index_finger_tip_x, index_finger_tip_y)

                    # 좌클릭 이벤트를 위한 엄지 좌표 값 + 중지 좌표 값
                    thumb_tip_x = int(thumb_tip.x * screen_w)
                    thumb_tip_y = int(thumb_tip.y * screen_h)
                    middle_finger_tip_x = int(middle_finger_tip.x * screen_w)
                    middle_finger_tip_y = int(middle_finger_tip.y * screen_h)

                    distance = ((thumb_tip_x - middle_finger_tip_x) ** 2 + (
                                thumb_tip_y - middle_finger_tip_y) ** 2) ** 0.5

                    if distance < 50:
                        pyautogui.leftClick()

                    # 우클릭 이벤트를 위한 약지 좌표 값
                    ring_finger_tip_x = int(ring_finger_tip.x * screen_w)
                    ring_finger_tip_y = int(ring_finger_tip.y * screen_h)

                    distance = ((thumb_tip_x - ring_finger_tip_x) ** 2 + (thumb_tip_y - ring_finger_tip_y) ** 2) ** 0.5

                    if distance < 50:
                        pyautogui.rightClick()

                # 왼손(키보드)
                elif handedness.classification[0].label == 'Left':

                    '''
                    type: 사용하는 손가락 수에 따른 타입
                    point_1: 기준점이 되는 포인트
                    point_2: 이동하는 포인트
                    input: 입력해야 하는 이벤트
                    output: 출력해야 하는 이벤트
                    detail: 출력하는 이벤트의 상세 정보
                    
                    type: '1 finger'|'2 finger'|'3 finger'|'4 finger'|'5 finger'
                    point_1: [str]
                    point_2: [str]
                    input: ['cross' | 'over' | 'under']
                    output: 'press' | 'click' | 'hold'
                    detail: 'right' | 'left' | keyboard_key
                    '''
                    pass
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) != -1:
            break


if __name__ == '__main__':
    __main__()
    # app.run()
