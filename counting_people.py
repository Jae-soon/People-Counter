import cv2 as cv
import numpy as np
import argparse
import os, sys
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
import time

confThreshold = 0.6  
nmsThreshold = 0.5   
inpWidth = 512      
inpHeight = 512     
# inpWidth = 416      
# inpHeight = 416    
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')

parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# modelConfiguration = "yolov3-spp.cfg";
# modelWeights = "yolov3-spp.weights";
modelConfiguration = "yolov4.cfg";
modelWeights = "yolov4.weights";
# modelConfiguration = "yolov4-csp.cfg";
# modelWeights = "yolov4-csp.weights";

print("[INFO] loading model...")
net = cv.dnn.readNet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

writer = None
W = None
H = None
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
 
totalDown = 0
totalUp = 0

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    # cv.line(frame, (0, frameHeight//2 - 50), (frameWidth, frameHeight//2 - 50), (0, 255, 255), 2)

    cv.line(frame, (frameWidth//2, 0), (frameWidth//2, frameHeight), (0, 255, 255), 2)

    cv.circle(frame,(left+(right-left)//2, top+(bottom-top)//2), 3, (0,0,255), -1)
        

    counter = []
    if (top+(bottom-top)//2 in range(frameHeight//2 - 2,frameHeight//2 + 2)):
        coun +=1

        counter.append(coun)

    label = 'Pedestrians: '.format(str(counter))
    cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(int(classId))
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            objects = ct.update(rects)
            counting(objects)


def counting(objects):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp


    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
 
        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
 
            # if not to.counted:
            #     if direction < 0 and centroid[1] in range(frameHeight//2 - 50, frameHeight//2 + 50):
            #         totalUp += 1
            #         to.counted = True
            #     elif direction > 0 and centroid[1] in range(frameHeight//2 - 50, frameHeight//2 + 50):
            #         totalDown += 1
            #         to.counted = True
            if not to.counted:
                if direction < 0 and centroid[0] in range(frameWidth//2 - 50, frameWidth//2 + 50):
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and centroid[0] in range(frameWidth//2 - 50, frameWidth//2 + 50):
                    totalDown += 1
                    to.counted = True

        trackableObjects[objectID] = to
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv.putText(frame, text, (10, frameHeight - ((i * 40) + 20)),
            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"

if (args.video):
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo4_out_py.avi'
else:
    cap = cv.VideoCapture(0)
    hasFrame, frame = cap.read()
    
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (800,570))

while cv.waitKey(1) < 0:
    start_time = time.process_time()   
    hasFrame, frame = cap.read()
    # frame = frame[300:1100, : 600]
    frame = frame[150:, 200:1000]
    # print(frame.shape)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
    cv.line(frame, (frameWidth//2, 0), (frameWidth//2, frameHeight), (0, 255, 255), 2)

    
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        cap.release()
        break

    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)

    t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    vid_writer.write(frame.astype(np.uint8))
    end_time = time.process_time()
    print(f"time elapsed : {int(round((end_time - start_time) * 1000))}ms")
    cv.imshow(winName, frame)