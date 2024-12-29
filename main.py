from flask import Flask, jsonify, request

import cv2
import mediapipe
import pyautogui

from mediapipe.python.solutions.hands import HandLandmark

app = Flask(__name__)

press_ = {'q': True, 'w': True, 'e': True, 'r': True}
click_ = {'left': True, 'right': True}


@app.route('/landmarks', methods=['GET'])
def get_landmarks():
    hand_landmark_dict = {key: value.value for key, value in HandLandmark.__members__.items()}
    return jsonify(hand_landmark_dict)


@app.route('/event', methods=['GET'])
def get_events():
    hand_landmark_dict = {key: value.value for key, value in HandLandmark.__members__.items()}
    return jsonify(hand_landmark_dict)


class Event:
    def __init__(self, point_1: float, point_2: float, key: str):
        self.point_1 = point_1
        self.point_2 = point_2
        self.key = key


def __main__():
    global press_, click_
    mp_hands = mediapipe.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mediapipe.solutions.drawing_utils

    screen_w, screen_h = pyautogui.size()

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

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                ring_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
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
                    trigger_ = distance < 50
                    if trigger_ and click_['left']:
                        pyautogui.leftClick()
                        click_['left'] = False
                    elif not trigger_ and not click_['left']:
                        click_['left'] = True

                    # 우클릭 이벤트를 위한 약지 좌표 값
                    ring_finger_tip_x = int(ring_finger_tip.x * screen_w)
                    ring_finger_tip_y = int(ring_finger_tip.y * screen_h)

                    distance = ((thumb_tip_x - ring_finger_tip_x) ** 2 + (thumb_tip_y - ring_finger_tip_y) ** 2) ** 0.5

                    trigger_ = distance < 50
                    if trigger_ and click_['right']:
                        pyautogui.rightClick()
                        click_['right'] = False
                    elif not trigger_ and not click_['right']:
                        click_['right'] = True

                # 왼손(키보드)
                elif handedness.classification[0].label == 'Left':
                    index_finger_mcp_y = int(index_finger_mcp.y * screen_h)
                    middle_finger_mcp_y = int(middle_finger_mcp.y * screen_h)
                    ring_finger_mcp_y = int(ring_finger_mcp.y * screen_h)
                    pinky_mcp_y = int(pinky_mcp.y * screen_h)

                    index_finger_tip_y = int(index_finger_tip.y * screen_h)
                    middle_finger_tip_y = int(middle_finger_tip.y * screen_h)
                    ring_finger_tip_y = int(ring_finger_tip.y * screen_h)
                    pinky_tip_y = int(pinky_tip.y * screen_h)

                    trigger_ = index_finger_mcp_y < index_finger_tip_y
                    if trigger_ and press_['q']:
                        pyautogui.press('q')
                        press_['q'] = False
                    elif not trigger_ and not press_['q']:
                        press_['q'] = True

                    trigger_ = middle_finger_mcp_y < middle_finger_tip_y
                    if trigger_ and press_['w']:
                        pyautogui.press('w')
                        press_['w'] = False
                    elif not trigger_ and not press_['w']:
                        press_['w'] = True

                    trigger_ = ring_finger_mcp_y < ring_finger_tip_y
                    if trigger_ and press_['e']:
                        pyautogui.press('e')
                        press_['e'] = False
                    elif not trigger_ and not press_['e']:
                        press_['e'] = True

                    trigger_ = pinky_mcp_y < pinky_tip_y
                    if trigger_ and press_['r']:
                        pyautogui.press('r')
                        press_['r'] = False
                    elif not trigger_ and not press_['r']:
                        press_['r'] = True

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

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    __main__()
    # app.run()
