    
import tensorflow as tf
from os import path
import numpy as np
from scipy import misc
from styx_msgs.msg import TrafficLight
import cv2
import rospy

class TLColorDetectorDL(object):
    def __init__(self, debug, model_path):
        self.debug = debug

    def predict(self, image):
        pass