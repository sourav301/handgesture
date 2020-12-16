# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:23:12 2020

@author: SOURAV
""" 
import PIL
from PIL import ImageTk, Image
from tkinter import *
import tkinter 
import cv2 
import os
import time 
folderpath = "images"
root = tkinter.Tk()
root.geometry("500x500")
f = Frame()
f.pack(side=BOTTOM)


def captureImage():
    i=1
    vid = cv2.VideoCapture(0)
    while i<=5:
        ret, frame = vid.read() 
        
        folderName = os.path.join(folderpath,E1.get())
        if not os.path.exists(folderName):
            os.mkdir(folderName)
        filename = os.path.join(folderName,str(i)+".jpg")
        cv2.imwrite(filename, frame)  
        var.set(str(i)+".jpg saved")
        i+=1
        root.update()
        time.sleep(1)
    vid.release() 

E1 = Entry(f)
E1.delete(0,END)
E1.insert(0,"1") 
E1.pack(side = LEFT)

B = tkinter.Button(f, text ="Capture", command = captureImage)
B.pack(side=LEFT)

var = StringVar()
var.set('Ready')
L1 = Label(f,textvariable = var)
L1.pack(side=LEFT)
canvas = Canvas(root, width = 300, height = 300, bg="black")      
canvas.pack(side=TOP) 

def checkImage():
    i=1
    while i<=5: 
        
        folderName = os.path.join(folderpath,E1.get())
        if not os.path.exists(folderName):
            return
        filename = os.path.join(folderName,str(i)+".jpg") 
        print(filename) 
        img = PIL.Image.open(filename)
        w,h = img.size 
        img = ImageTk.PhotoImage(img.resize((w//2,h//2)) ) 
        canvas.create_image(100,100, image=img)    
        i+=1
        root.update()
        time.sleep(1)
    

B = tkinter.Button(f, text ="Check", command = checkImage)
B.pack()




root.mainloop()