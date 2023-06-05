import cv2
import numpy as np
import imutils

protopath = "MobileNetSSD_deploy.prototxt.txt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def func1(input):
    image = cv2.imread(input)
    image = imutils.resize(image, width=600)

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          [idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [idx], 2)

    cv2.imshow("Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    func1('people_images\people1.jpg')
    func1('people_images\people2.jpg')
    func1('people_images\people3.jpg')
    func1('people_images\people4.jpg')
    func1('people_images\people5.jpg')


main()
