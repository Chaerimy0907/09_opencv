# EAR (Eye Aspect Ratio) : 눈 깜빡임을 감지하기 위해 사용하는 값

import cv2
import dlib
from scipy.spatial import distance

# EAR 계산 함수
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A+B) / (2.0 * C)
    return ear

# 눈 인덱스
LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]

# 눈 좌표 추출 함수
def get_eye_points(landmarks, eye_indices):
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 카메라 연결
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 눈 좌표
        left_eye = get_eye_points(landmarks, LEFT_EYE_IDX)
        right_eye = get_eye_points(landmarks, RIGHT_EYE_IDX)

        # EAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # EAR 출력
        cv2.putText(img, f"EAR : {avg_ear : .2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    cv2.imshow("EAR Calculation", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()