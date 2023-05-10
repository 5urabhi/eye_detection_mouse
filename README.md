
--------------------------------------------------------------------------------------------

Logic flow:

- find face and track the right eye;
- calibrate the proper range of allowed space for (comfortable) movement;

(then, frame by frame)

- track the right eye and project its centroid on a virtual keyboard in a "working window";
- take inputs and, specifically, "press" the key, using blinking detection;
- updtate a text string and print on screen (and, by just adding a couple of line, on a file).

--------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------
Requirements: 
- python2 (python3 works too)
- opencv
- numpy
- dlib (and the trained model shape_predictor_68_face_landmarks.dat; see below)
--------------------------------------------------------------------------------------------
To install dlib use this link https://pypi.org/simple/dlib/
download latest and then extract and then open setup.py you will get commands for the running
you will also need cmake so run command "pip install cmake" beforehand.

also run "eye_key.py" that is the main code.

i combined following codes: 

Facial mapping (landmarks) with Dlib + python can be found here:
https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

The trained model file shape_predictor_68_face_landmarks.dat can be found here:
https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
and all the details here:
https://github.com/davisking/dlib-models
