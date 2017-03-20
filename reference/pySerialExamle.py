import serial
import numpy as np
import pandas as pd


with serial.Serial('/dev/cu.usbmodem1411', 9600, timeout=10) as ser:
    line = ser.readline()
    while(line!=""):
        line = ser.readline()
        print(line)
