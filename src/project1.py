import cv2
import numpy as np
import os, glob, csv
from datetime import datetime

# 변수 설정
base_dir = '../faces'
min_accuracy = 85
attendance_file = '출석.csv'
attended = set()   # 출석한 사람들 기록

# LBP 얼굴 인식기 및 케스케이드 얼굴 검출기 생성 및 훈련 모델 읽기
face_classifier = cv2.CascadeClassifier(\
                '../data/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(base_dir, 'all_face.xml'))

# 디렉토리 이름으로 사용자 이름과 아이디 매핑 정보 생성
dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]
names = dict([])
for dir in dirs:
    dir = os.path.basename(dir)
    name, id = dir.split('_')
    names[int(id)] = name

# 출석 기록용 파일 생성
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['이름', '출석시간'])

# 카메라 캡처 장치 준비 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("no frame")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 얼굴 검출
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # 얼굴 영역 표시하고 샘플과 같은 크기로 축소
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # LBP 얼굴 인식기로 예측
        label, confidence = model.predict(face)
        if confidence < 400:
            # 정확도 거리를 퍼센트로 변환
            accuracy = int( 100 * (1 -confidence/400))
            if accuracy >= min_accuracy:
                msg =  '%s(%.0f%%)'%(names[label], accuracy)

                # 이미 출석했는지 확인
                if name not in attended:
                    attended.add(name)
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(attendance_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, now])
                    print(f'{name} 출석 완료 : {now}')
                    msg = f'{name} V'
                else:
                    msg = f'{name} (Already Attendance)'

            else:
                msg = 'Unknown'

        # 사용자 이름과 정확도 결과 출력
        txt, base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 1, 3)
        cv2.rectangle(frame, (x,y-base-txt[1]), (x+txt[0], y+txt[1]), \
                    (0,255,255), -1)
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, \
                    (200,200,200), 2,cv2.LINE_AA)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == 27: #esc 
        break
    
cap.release()
cv2.destroyAllWindows()     