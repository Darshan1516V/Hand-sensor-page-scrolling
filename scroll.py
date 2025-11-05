import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe and PyAutoGUI
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure hand detection
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

def is_palm_open(landmarks):
    # Open palm: fingers are extended above the MCP joints
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    open_fingers = 0

    for tip, mcp in zip(finger_tips, finger_mcp):
        if landmarks[tip].y < landmarks[mcp].y:
            open_fingers += 1
    
    # If 4 fingers are open, we assume palm is open
    return open_fingers >= 4

prev_state = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture detection
            if is_palm_open(hand_landmarks.landmark):
                current_state = "open"
            else:
                current_state = "closed"

            if current_state != prev_state:
                if current_state == "open":
                    print("Scrolling Up")
                    pyautogui.scroll(300)  # Scroll up
                elif current_state == "closed":
                    print("Scrolling Down")
                    pyautogui.scroll(-300)  # Scroll down
                prev_state = current_state

    cv2.imshow("Palm Gesture Scrolling", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
