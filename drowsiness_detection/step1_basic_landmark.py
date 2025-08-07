# 기본 랜드마크 검출

import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 카메라 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame')
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray)
    for face in faces:
        # 얼굴 영역을 좌표로 변환 후 사각형 표시
        x,y = face.left(), face.top()
        w,h = face.right()-x, face.bottom()-y
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0),1)

        # 랜드마크 예측
        landmarks = predictor(gray, face)
        for i in range(68):
            # 부위별 좌표 추출 및 표시
            part = landmarks.part(i)
            cv2.circle(img, (part.x, part.y), 2, (0,0,255), -1)

    cv2.imshow("face landmark", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()