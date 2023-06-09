import cv2
import numpy as np
import os

CUR_DIR = os.path.abspath('.')
yolov3_weights_path = os.path.join(CUR_DIR, 'detection/yolov3.weights')
yolov3_config_path = os.path.join(CUR_DIR, 'detection/yolov3.cfg')
coco_path = os.path.join(CUR_DIR, 'detection/coco.names')

# Yolo 로드
net = cv2.dnn.readNet(yolov3_weights_path, yolov3_config_path)
classes = []
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 0, size=(len(classes), 3))
threshold = 0.5

def detect(img, file_path, ext):
    # 이미지 가져오기
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.25)
    idx = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if x < 0 or y < 0:
                continue
            label = str(classes[class_ids[i]])
            obj = img[y:y+h, x:x+w]
            image_name = f'{label}_{idx}.{ext}'
            cv2.imwrite(os.path.join(file_path, image_name), obj)

            idx += 1
