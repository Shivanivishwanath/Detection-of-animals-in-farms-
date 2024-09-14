import numpy as np
import time
import cv2
import os
from playsound import playsound
import threading
from twilio.rest import Client

# Your Twilio Account SID and Auth Token
account_sid = '...............................'
auth_token = '................................'

# Your Twilio phone number
twilio_phone_number = '+............'

# Recipient phone number
recipient_phone_number = '+91.........'

def alert():
    threading.Thread(target=playsound, args=('alarm.wav',), daemon=True).start()

def send_sms(label):
    client = Client(account_sid, auth_token)
    body = f'{label} has been detected'
    message = client.messages.create(
        body=body,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )

    print("SMS sent successfully.")

def async_sms(label):
    threading.Thread(target=send_sms, args=(label,), daemon=True).start()

args = {"confidence": 0.5, "threshold": 0.3}
flag = False

labelsPath = "./yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
final_classes = ['bird', 'cat', 'dog', 'sheep', 'horse', 'cow', 'elephant', 'zebra', 'bear', 'giraffe']

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

weightsPath = os.path.abspath("./yolo-coco/yolov3-tiny.weights")
configPath = os.path.abspath("./yolo-coco/yolov3-tiny.cfg")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = net.getUnconnectedOutLayersNames()

vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)

flag = True

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if (LABELS[classIDs[i]] in final_classes):
                if (flag):
                    alert()
                    flag = False
                    async_sms(LABELS[classIDs[i]])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                            confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    else:
        flag = True

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
