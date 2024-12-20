import cv2
import mediapipe
import pyautogui


def __main__():
    print('pjt start')

    mp_hands = mediapipe.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mediapipe.solutions.drawing_utils

    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gum_gi = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                jung_ji = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                index_x, index_y = int(gum_gi.x * screen_w), int(gum_gi.y * screen_h)
                middle_x, middle_y = int(jung_ji.x * screen_w), int(jung_ji.y * screen_h)

                distance = ((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2) ** 0.5

                if distance < 50:
                    pyautogui.moveTo(index_x, index_y)

                    cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) != -1:
            break


if __name__ == '__main__':
    __main__()
