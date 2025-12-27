import cv2
import mediapipe as mp
import time
import random
from collections import Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== COLOR CONFIG (PASTEL) ==================
PASTEL_BLUE = (255, 200, 180)   # biru muda pastel (BGR)
WHITE = (255, 255, 255)
GREEN = (120, 255, 180)
RED = (180, 180, 255)
YELLOW = (200, 255, 255)

# ================== HAND CONNECTIONS ==================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ================== INIT MEDIAPIPE TASKS ==================
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ================== GAME CONFIG ==================
GESTURES = ["Rock", "Paper", "Scissors"]

state = "WAITING"
buffer_gestures = []
buffer_duration = 2
countdown_duration = 3
result_duration = 2

user_choice = None
ai_choice = None
winner_text = ""

user_score = 0
ai_score = 0

state_start_time = time.time()

# ================== GESTURE DETECTION ==================
def detect_gesture(landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        fingers.append(landmarks[tip].y < landmarks[tip - 2].y)

    if fingers == [False, False, False, False]:
        return "Rock"
    elif fingers == [True, True, True, True]:
        return "Paper"
    elif fingers == [True, True, False, False]:
        return "Scissors"
    return None

# ================== GAME LOGIC ==================
def determine_winner(user, ai):
    if user == ai:
        return "Draw"
    elif (user == "Rock" and ai == "Scissors") or \
         (user == "Paper" and ai == "Rock") or \
         (user == "Scissors" and ai == "Paper"):
        return "User Wins"
    else:
        return "AI Wins"

# ================== MAIN LOOP ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)
    current_gesture = None

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        current_gesture = detect_gesture(landmarks)

        h, w, _ = frame.shape
        points = []

        # ===== Draw pastel landmarks =====
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 4, PASTEL_BLUE, -1)

        # ===== Draw pastel connections =====
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], PASTEL_BLUE, 2)

    now = time.time()

    # ================== STATE MACHINE ==================
    if state == "WAITING":
        buffer_gestures.clear()
        user_choice = None
        ai_choice = None
        winner_text = "Show your hand ✋"
        if current_gesture:
            state = "READY"
            state_start_time = now

    elif state == "READY":
        winner_text = "Hold steady..."
        if current_gesture:
            buffer_gestures.append(current_gesture)

        if now - state_start_time >= buffer_duration:
            if buffer_gestures:
                user_choice = Counter(buffer_gestures).most_common(1)[0][0]
                state = "COUNTDOWN"
                state_start_time = now
            else:
                state = "WAITING"

    elif state == "COUNTDOWN":
        remaining = countdown_duration - int(now - state_start_time)
        winner_text = f"Get Ready: {remaining}"
        if remaining <= 0:
            ai_choice = random.choice(GESTURES)
            winner_text = determine_winner(user_choice, ai_choice)

            if winner_text == "User Wins":
                user_score += 1
            elif winner_text == "AI Wins":
                ai_score += 1

            state = "RESULT"
            state_start_time = now

    elif state == "RESULT":
        if now - state_start_time >= result_duration:
            state = "WAITING"

    # ================== UI ==================
    cv2.putText(frame, f"State: {state}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

    cv2.putText(frame, f"User: {user_choice}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)

    cv2.putText(frame, f"AI: {ai_choice}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)

    cv2.putText(frame, f"Result: {winner_text}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)

    cv2.putText(frame, f"Score {user_score} : {ai_score}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

    cv2.imshow("Gesture Game - Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()