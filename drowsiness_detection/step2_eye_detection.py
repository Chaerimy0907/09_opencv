# dlib 랜드마크 중 눈 영역만 추출

import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 눈 랜드마드 인덱스
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# 카메라 설정
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 눈 좌표 리스트
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_IDX]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_IDX]

        # 눈 테두리 시각화
        cv2.polylines(img, [np.array(left_eye, dtype=np.int32)], isClosed=True, color=(0,255,0), thickness=1)
        cv2.polylines(img, [np.array(right_eye, dtype=np.int32)], isClosed=True, color=(0,255,0), thickness=1)

    cv2.imshow('Step2 : Eye Detection', img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()