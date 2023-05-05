import cv2
import mediapipe as mp
cam = cv2. VideoCapture (0)
face_mesh = mp.solutions.face_mesh.FaceMesh()#refine_landmarks=True)
while True:
    success, image = cam.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(image)
    landmark_points = output.multi_face_landmarks
    frame_h,frame_w,_=image.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for landmark in landmarks:
            x=int(landmark.x)#*frame_w)
            y=int(landmark.y)#*frame_h)
            cv2.circle(image, (x, y), 3, (0, 255, 0))
            print(landmark_points)
    cv2. imshow ('Eye Controlled Mouse', image)