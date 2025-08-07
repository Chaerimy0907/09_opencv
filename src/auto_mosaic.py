# 카메라로 찍은 영상에서 얼굴 부분을 자동으로 모자이크 처리하는 프로그램
# haarcascade로 얼굴 검출 -> 모자이크

import cv2

# 변수
rate = 15

# 얼굴과  검출을 위한 케스케이드 분류기 생성 
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():    
    ret, frame = cap.read()  # 프레임 읽기
    if ret:
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 검출    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, \
                                        minNeighbors=5, minSize=(80,80))
        if len(faces) == 1:
            (x,y,w,h) = faces[0]
            # 얼굴 영역 표시
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),1)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (w//rate, h//rate))

            roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)
            img[y:y+h, x:x+w] = roi
            cv2.imshow('Face Mosaic', img)
    else:
        break
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()