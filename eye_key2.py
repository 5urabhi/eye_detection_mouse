from __future__ import print_function

import time
import cv2
import numpy as np
import random
import dlib
import sys
import pyautogui

from eye_key_funcs import *
from projected_keyboard import get_keyboard

# ------------------------------------ Inputs
camera_ID = 0  # select webcam

width_keyboard = 1000
height_keyboard = 500  # [pixels]
offset_keyboard = (100, 80)  # pixel offset (x, y) of keyboard coordinates

resize_eye_frame = 4.5  # scaling factor for window's size
resize_frame = 0.3  # scaling factor for window's size
# ------------------------------------

# ------------------------------------------------------------------- INITIALIZATION
# Initialize the camera
camera = init_camera(camera_ID=camera_ID)

# take size screen
size_screen = (camera.get(cv2.CAP_PROP_FRAME_HEIGHT), camera.get(cv2.CAP_PROP_FRAME_WIDTH))

# make a page (2D frame) to write & project
keyboard_page = make_black_page(size=size_screen)
calibration_page = make_black_page(size=size_screen)

# Initialize keyboard
key_points = get_keyboard(width_keyboard=width_keyboard,
                          height_keyboard=height_keyboard,
                          offset_keyboard=offset_keyboard)

# upload face/eyes predictors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# -------------------------------------------------------------------

# ------------------------------------------------------------------- CALIBRATION
corners = [(128, 90), (384, 90), (640, 90), (896, 90), (1152, 90),
           (128, 270), (384, 270), (640, 270), (896, 270), (1152, 270),
           (128, 450), (384, 450), (640, 450), (896, 450), (1152, 450),
           (128, 630), (384, 630), (640, 630), (896, 630), (1152, 630)]

calibration_cut = []
corner = 0

while corner < 20:  # calibration of 4 corners
    ret, frame = camera.read()  # Capture frame
    frame = adjust_frame(frame)  # rotate / flip
    frame_h, frame_w = frame.shape[:2]

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray-scale to work with

    cv2.putText(calibration_page, 'calibration: look at the circle and blink',
                tuple((np.array(size_screen) / 7).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.circle(calibration_page, corners[corner], 40, (0, 255, 0), -1)

    faces = detector(gray_scale_frame)
    if len(faces) > 1:
        print('please avoid multiple faces.')
        sys.exit()

    for face in faces:
        display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

        landmarks = predictor(gray_scale_frame, face)
        display_face_points(frame, landmarks, [0, 68], color='red')

        right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
        print(right_eye_coordinates)
        display_eye_lines(frame, right_eye_coordinates, 'green')
        # define the coordinates of the pupil from the centroid of the right eye
        pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis=0).astype('int')

        if is_blinking(right_eye_coordinates):
            calibration_cut.append(pupil_coordinates)

            cv2.putText(calibration_page, 'ok',
                        tuple(np.array(corners[corner]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            time.sleep(0.3)
            corner += 1

    show_window('projection', calibration_page)
    show_window('frame', cv2.resize(frame, (640, 360)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# -------------------------------------------------------------------

# ------------------------------------------------------------------- PROCESS CALIBRATION
x_cut_min, x_cut_max, y_cut_min, y_cut_max = find_cut_limits(calibration_cut)
offset_calibrated_cut = [x_cut_min, y_cut_min]

print('message for user')
cv2.putText(calibration_page, 'calibration done. please wait for the keyboard...',
            tuple((np.array(size_screen) / 5).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
show_window('projection', calibration_page)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('keyboard appearing')
# -------------------------------------------------------------------

# ------------------------------------------------------------------- WRITING
pressed_key = True
string_to_write = "text: "

while True:
    ret, frame = camera.read()  # Capture frame
    frame = adjust_frame(frame)  # rotate / flip

    cut_frame = np.copy(frame[y_cut_min:y_cut_max, x_cut_min:x_cut_max])

    keyboard_page = make_black_page(size=size_screen)
    dysplay_keyboard(img=keyboard_page, keys=key_points)
    text_page = make_white_page(size=(200, 800))

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray-scale to work with

    faces = detector(gray_scale_frame)
    if len(faces) > 1:
        print('please avoid multiple faces..')
        sys.exit()

    for face in faces:
        display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

        landmarks = predictor(gray_scale_frame, face)
        display_face_points(frame, landmarks, [0, 68], color='red')

        right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
        display_eye_lines(frame, right_eye_coordinates, 'green')

    pupil_on_frame = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis=0).astype('int')
    pupil_on_cut = np.array(
        [pupil_on_frame[0] - offset_calibrated_cut[0], pupil_on_frame[1] - offset_calibrated_cut[1]])
    cv2.circle(cut_frame, (pupil_on_cut[0], pupil_on_cut[1]), int(take_radius_eye(right_eye_coordinates) / 1.5),
               (255, 0, 0), 3)

    if pupil_on_cut_valid(pupil_on_cut, cut_frame):
        pupil_on_screen = (int(pupil_on_cut[0] * resize_frame), int(pupil_on_cut[1] * resize_frame))
        pyautogui.moveTo(pupil_on_screen[0], pupil_on_screen                          [1], duration=0.1)

    show_window('frame', cv2.resize(frame, (640, 360)))
    show_window('cut frame', cv2.resize(cut_frame, (640, 360)))

    # Detect key press
    key_pressed = identify_key(key_points=key_points, coordinate_X=pupil_on_cut[1], coordinate_Y=pupil_on_cut[0])

    #key_pressed = detect_key_press(pupil_on_cut, key_points)

    if key_pressed is not None:
        if key_pressed == 'enter':
            print('Text entered:', string_to_write)
            string_to_write = "text: "
        elif key_pressed == 'backspace':
            string_to_write = string_to_write[:-1]
        elif key_pressed == 'space':
            string_to_write += ' '
        elif isinstance(key_pressed, str):  # Check if key_pressed is a string
            string_to_write += key_pressed

    cv2.putText(text_page, string_to_write, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    show_window('text', text_page)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()