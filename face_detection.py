import dlib
import cv2
import time
import imutils
from imutils import face_utils

EAR_THRESHOLD = 0.3
FRAMES_IN_A_ROW = 2


def compute_ear(eye_marks):
    a = sum(abs(eye_marks[1] - eye_marks[5]))
    b = sum(abs(eye_marks[2] - eye_marks[4]))
    c = sum(2 * abs(eye_marks[0] - eye_marks[3]))
    ear = (a+b)/c

    return round(ear, 2)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
time.sleep(2.0)

frames_counter = 0
total_blinks = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1280)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]

        ear_left = compute_ear(left_eye)
        ear_right = compute_ear(right_eye)

        ear_avr = round((ear_left + ear_right) / 2, 3)

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)

        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)

        if ear_avr < EAR_THRESHOLD:
            frames_counter += 1
        else:
            if frames_counter >= FRAMES_IN_A_ROW:
                total_blinks += 1
            frames_counter = 0

        for (lx, ly), (rx, ry) in zip(left_eye, right_eye):
            cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
            cv2.circle(frame, (rx, ry), 2, (0, 0, 255), -1)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"E. A. R.: {ear_avr}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Blink count: {total_blinks}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
