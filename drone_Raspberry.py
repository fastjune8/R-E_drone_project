import cv2
import numpy as np
from machine import UART
import time

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 카메라 설정
cap = cv2.VideoCapture(0)

# 드론 제어 초기화
uart0 = UART(0, 9600, bits=8, parity=None, stop=1)

# 드론 제어 명령 전송 함수
def sendDroneCommand(roll, pitch, yaw, throttle):
    startBit_1, startBit_2, startBit_3, startBit_4 = 0x26, 0xa8, 0x14, 0xb1
    leng = 0x14
    option, p_vel, y_vel = 0x000f, 0x0064, 0x0064
    payLoad = [0] * 14

    # 드론 제어 값 설정
    payLoad[0] = roll & 0x00ff
    payLoad[1] = (roll >> 8) & 0x00ff
    payLoad[2] = pitch & 0x00ff
    payLoad[3] = (pitch >> 8) & 0x00ff
    payLoad[4] = yaw & 0x00ff
    payLoad[5] = (yaw >> 8) & 0x00ff
    payLoad[6] = throttle & 0x00ff
    payLoad[7] = (throttle >> 8) & 0x00ff
    payLoad[8] = option & 0x00ff
    payLoad[9] = (option >> 8) & 0x00ff
    payLoad[10] = p_vel & 0x00ff
    payLoad[11] = (p_vel >> 8) & 0x00ff
    payLoad[12] = y_vel & 0x00ff
    payLoad[13] = (y_vel >> 8) & 0x00ff

    # 체크섬 계산
    checkSum = sum(payLoad) & 0x00ff

    # 명령 전송
    uart0.write("at+writeh000d".encode())
    uart0.write((hex(startBit_1)[2:4]).encode())
    uart0.write((hex(startBit_2)[2:4]).encode())
    uart0.write((hex(startBit_3)[2:4]).encode())
    uart0.write((hex(startBit_4)[2:4]).encode())
    uart0.write((hex(leng)[2:4]).encode())
    uart0.write((hex(checkSum)[2:4]).encode())

    for i in range(14):
        if payLoad[i] < 0x10:
            uart0.write(('0'+hex(payLoad[i])[2:4]).encode())
        else:
            uart0.write((hex(payLoad[i])[2:4]).encode())

    uart0.write("\r".encode())
    time.sleep(0.1)

# 메인 루프
while True:
    # 카메라에서 프레임 읽기
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # YOLO 모델로 물체 감지
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 물체 인식 결과 처리
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # 물체의 중심 좌표 계산
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                # 드론 제어 로직
                roll = (center_x - width / 2) / width * 100
                pitch = (center_y - height / 2) / height * 100
                sendDroneCommand(roll, pitch, 0, 100)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()