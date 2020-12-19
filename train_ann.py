# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:47:42 2020

@author: SIBSANKAR
"""

from sklearn import svm
from sklearn import preprocessing
from tensorflow.keras import layers,models 
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
f_multi_dimen= np.load("data_features.npy") #n*21*2 dimen
o= np.load("data_output.npy") 
f = f_multi_dimen.reshape((len(o),-1))  

clf = svm.SVC()
clf.fit(f, o)


label_encoder = preprocessing.LabelEncoder()
o = label_encoder.fit_transform(o)
# Train SVM
# clf = svm.SVC()
# clf.fit(f, o)

# Train NN
ann_model = models.Sequential()
# ann_model.Add(layers.Dense(21),activation='relu',input_shape=(21,2))
ann_model.add(layers.Flatten())
ann_model.add(layers.Dense(21))
ann_model.add(layers.Dense(21))
ann_model.add(layers.Dense(10))

ann_model.compile(optimizer='adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
history= ann_model.fit(f_multi_dimen,o,epochs=100,validation_data=(f_multi_dimen,o))

ann_model.save("model/model_ann")
  
plt.plot( history.history['accuracy'] , label='accuracy')
plt.plot( history.history['val_accuracy'] , label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend(loc='lower right')

test_loss, test_acc = ann_model.evaluate(f_multi_dimen,o,verbose=2)
print(np.argmax(ann_model.predict(f_multi_dimen),axis=1))
