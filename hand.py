import cv2 
import os
protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_102000.caffemodel"
nPoints = 22

frame = cv2.imread("C:\\Users\\SIBSANKAR\\Desktop\\dev\\hand.jpg")
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
cv2.imshow('Output-Keypoints', frame)


inHeight,inWidth,_ = frame.shape 

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
points = []
frameHeight,frameWidth,_=frame.shape
frameCopy=frame.copy()
threshold=.1 
for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if prob > threshold : 
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
    else :
        points.append(None)
 
POSE_PAIRS=[]
POSE_PAIRS.append((0,2))
POSE_PAIRS.append((2,4))
POSE_PAIRS.append((2,5))
POSE_PAIRS.append((5,8))
POSE_PAIRS.append((9,12))
POSE_PAIRS.append((13,16))
POSE_PAIRS.append((17,20))
# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

cv2.imshow('Output-Skeleton', frame)




cv2.imshow('Output-Keypoints', frameCopy)       
cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
#cv2.imshow('Output-Keypoints', frame)
  