# Learn Hand Gestures
Application that detects hand gesture.
The idea of the project is to create a python application which can be used to train a neural network to recognize hand symbols.

![Sample Video](https://github.com/sourav301/handgesture/blob/master/media/hand.gif)

A SVM is trained with 5 images of hand in the left, right and middle position.
The video shows that the SVM can correctly classify the hand position in a live webcam video.



## Features Extraction
The [link](https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/) descibes in detail to extract 21 points from a hand image. A deel learning model from CMU Perceptual Computing Lab has been used for this. Refer to [this paper](https://arxiv.org/pdf/1704.07809.pdf) for more details.

## All in one Tkinter app 
This app build using Tkinter has the following features

* Capture and label images
* Extract Features
* Train a SVM
* Prediction on live video


1. Capture and label images
Enter a label (ex. 'left') and click on capture. Position hand in front of the webcam and 5 images are captured. The images are stored in images folder with a subfolder named after the label.

1. Extract features and train
Clicking on the train button does 2 task. Firstly it loops over the stored images in the images folder and extracts features using the pretrained model. Secondly it trains a SVM with these extracted features.

1. Predict
When the predict button is clicked the app is ready to predict position of a hand when positioned in front of the webcam. The predicted label is displayed in the label.
