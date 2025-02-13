import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))


EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 50


cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)  # TODO usar fps para configurar quantos segundos eh o limite de olho fechado
frames_closed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0


        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            frames_closed += 1
            if frames_closed > FRAME_THRESHOLD:
                print("olhos fechados")
                cv2.putText(frame, "Olhos Fechados!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frames_closed = 0

    cv2.imshow("Deteccao de Olhos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
