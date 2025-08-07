'''
1. EAR 임계값 (EAR_THRESHOLD) 지정
2. EAR이 임계값 이하일 때마다 frame_counter 증가
3. 일정 프레임 수 (CONSEC_FRAMES) 이상 지속되면 졸음 판단
4. 화면에 빨간 경고 메시지 출력 

calculate_ear(), get_eye_points(), LEFT_EYE_IDX, RIGHT_EYE_IDX 등은
이미 step3_ear_calculation.py 에 정의되어 있음
'''

import cv2
import dlib

# 졸음 감지 기준값
EAR_THRESHOLD = 0.25    # 눈을 감았다고 판단하는 EAR 기준값
CONSEC_FRAMES = 20      # 몇 프레임 이상 감겨야 졸음으로 판단할지
frame_counter = 0       # 눈 감은 상태 유지 프레임 수

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

        left_eye = get_eye_points(landmarks, LEFT_EYE_IDX)
        right_eye = get_eye_points(landmarks, RIGHT_EYE_IDX)

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_eye + right_eye) / 2.0

        # EAR 값 출력
        cv2.putText(img, f"EAR : {avg_ear : .2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 졸음 감지
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                cv2.putText(img, "Drowsiness Alert", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            frame_counter = 0

    cv2.imshow("Drowsiness Detection", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()