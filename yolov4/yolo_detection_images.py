import numpy as np
import cv2


def yolov3_detection(image_path):
    confidenceThreshold = 0.5
    NMSThreshold = 0.3
    modelConfiguration = 'cfg/yolov4-LP_licence.cfg'
    # don't forget to change this weights!!!
    modelWeights = 'weight/yolov4-LP_licence_final.weights'
    labelsPath = 'data/LP_licence.names'
    # modelConfiguration = 'cfg/yolov4-obj.cfg'
    # modelWeights = 'weight/yolov4-obj_11000.weights'
    # labelsPath = 'data/obj.names'

    labels = open(labelsPath).read().strip().split('\n')
    print(image_path)
    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    #image = cv2.imread('yolov3/images/good.jpeg')
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]
    # Determine output layer names
    layerName = net.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)
    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(
        boxes, confidences, confidenceThreshold, NMSThreshold)
    detection_Objects = []
    if(len(detectionNMS) > 0):
        detection_Objects = []
        file_name = image_path.split('.')[0]
        # with open(file_name+'.txt', 'w') as file:
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            area = w * h  # the area of the bounding box.
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            text = '{}: {:.4f},bbox:{}'.format(
                labels[classIDs[i]], confidences[i], area)
            cv2.putText(image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            detection_Objects.append(
                [labels[classIDs[i]], w*h, w, h, confidences[i]*100])

    print(detection_Objects)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)

    cv2.waitKey(0)
    return 'ok from yolo_detection_image~'

# for i in range(0,31):
#     try:
#         yolov3_detection('images/'+str(i)+'.jpg')
#     except:
#         pass


yolov3_detection('images/15.jpg')
