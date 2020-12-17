import cv2  

protoFile = "model\\pose_deploy.prototxt"
weightsFile = "model\\pose_iter_102000.caffemodel"
nPoints = 22

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile) 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

 
#cv2.imshow('Output-Keypoints', frame)
    
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100) 
    dsize = (width, height) 
    frame = cv2.resize(frame, dsize)

    inHeight,inWidth,_ = frame.shape 
    
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
  
    net.setInput(inpBlob)
    
    output = net.forward()
    points = []
    frameHeight,frameWidth,_=frame.shape
    frameCopy=frame.copy()
    threshold=.05
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
            cv2.line(frameCopy, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frameCopy, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    

    # Display the resulting frame 
    cv2.imshow('frame', frameCopy) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice  
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 