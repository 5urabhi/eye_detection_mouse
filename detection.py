import cv2
import mediapipe as mp
import pyautogui

try:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("Could not open camera")
except Exception as e:
    print(f"Error: {e}")
    exit()

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            print(id)
            if id == 0 :
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
        left = [landmarks[145], landmarks[159]]
        right=[landmarks[475],landmarks[477]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        for landmark in right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            print(left[0].y - left[1].y)
            pyautogui.leftClick()
            pyautogui.sleep(1)
        if (right[0].y - right[1].y) < 0.004:
            print(right[0].y - right[1].y)
            pyautogui.rightClick()
            pyautogui.sleep(1)
        if ((left[0].y - left[1].y) < 0.004) and ((right[0].y - right[1].y) < 0.004):
            #pyautogui.click()
            pyautogui.sleep(2)
    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)
