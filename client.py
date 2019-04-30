import cv2
import numpy as np
import socket

cap=cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8089))

# TODO add zlib compression - requires variable size messages
while True:
    ret,frame=cap.read()
    ### Assumes fixed size 2764800 bytes, (720, 1280, 3)
    clientsocket.sendall(frame.tobytes())
    print("Sent frame!!!")
