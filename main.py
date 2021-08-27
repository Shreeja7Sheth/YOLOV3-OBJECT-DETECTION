import cv2
import numpy as np

cap = cv2.VideoCapture("lambor.mp4")
# width height of the Target
whT = 320

# text file
classesFiles = "coco.names"
classNames = []

confThreshold = 0.5
nmsThreshold = 0.3
with open(classesFiles, 'rt') as f:
    # extracts info based on newline
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

# creating our network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# opencv as the backend and use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# find the probability and remove it if the confidence is less. If more, put it in the list
def findObjects(outputs, img):
    hT, wT, cT = img.shape  # height width and confidence
    bbox = []  # bb = x,y,width,height
    classIds = []  # classIDs
    confs = []  # confidence values

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            # find only the highest prob value
            classId = np.argmax(scores)
            confidence = scores[classId]

            # filter objects
            if confidence > confThreshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    print(len(bbox))
    # displays only the bb with max confidence and supresses other - maximum supression function
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    print(indices)
    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)


while True:
    # it will give us image and tell us whether it will be successfully retrieved or not
    success, img = cap.read()

    # convert image to blob as the network only accepts blob as input
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # names to refer the 3 types of output
    layerNames = net.getLayerNames()

    # displays the layer names
    # print(layerNames)

    # extract the output layer
    # we are getting the index of these outputs starting from 1
    # print(net.getUnconnectedOutLayers())

    # it gives the output as 3 different values of 3 different layers
    # for each layer, we want to get the value of 200 - to get the first element of i = 200 and subtract -1
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # gives the output names of our layers
    # print(outputNames)

    # Now we can send this image as a forward pass to the network
    # and we can find the output of these (['yolo_82', 'yolo_94', 'yolo_106']) 3 layers
    outputs = net.forward(outputNames)

    # elements are of class numpy.ndarray
    # output[0] = (300,85) = a matrix of 300 rows and 85 columns
    # we have 80 classes and we are getting 85 different values - why?

    # The first layer contains 300 bounding boxes(bb)
    # The second layer contains 1200 bb
    # The third layer contains 4800 bb
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    findObjects(outputs, img)

    cv2.imshow('Image', img)

    # delay for 1 millis
    cv2.waitKey(1)
