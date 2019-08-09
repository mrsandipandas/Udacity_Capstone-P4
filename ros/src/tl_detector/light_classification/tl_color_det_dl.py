    
import tensorflow as tf
from os import path
import numpy as np
from scipy import misc
from styx_msgs.msg import TrafficLight
import cv2
import matplotlib.pyplot as plt
import rospy
import tensorflow as tf

class TLColorDetectorDL(object):
    def __init__(self, debug, model_path):
        self.debug = debug

        self.sess = None
        self.path = model_path
        self.likelihood = 0.15
        self.coco_traffic_light_type = 10
        tf.reset_default_graph()
        gd = tf.GraphDef()
        gd.ParseFromString(tf.gfile.GFile(self.path, "rb").read())
        tf.import_graph_def(gd, name="object_detection_api")
        self.sess = tf.Session()
        
        g = tf.get_default_graph()
        self.image = g.get_tensor_by_name("object_detection_api/image_tensor:0")
        self.boxes = g.get_tensor_by_name("object_detection_api/detection_boxes:0")
        self.scores = g.get_tensor_by_name("object_detection_api/detection_scores:0")
        self.classes = g.get_tensor_by_name("object_detection_api/detection_classes:0")

    def predict(self, img):
        
        img_h, img_w = img.shape[:2]
        #rospy.loginfo("img_h,img_w is %s,%s",img_h,img_w)

        center_h = img_h // 2
        center_w = img_w // 2
        step_h = 150
        step_w = center_w - 150
        
        if self.debug:
            cv2.namedWindow('Debug', cv2.WINDOW_NORMAL)
            cv2.startWindowThread()

        for h0,w0 in [(center_h, center_w),(center_h-step_h, center_w),(center_h, center_w-step_w),(center_h-step_h, center_w-step_w),(center_h, center_w+step_w),(center_h-step_h, center_w+step_w)]:

           """
           The Model has 300x300 input image size.
           We pick 6 300x300 regions of interest manually from the incoming image.
           Loop through it until a traffic light is found
           """
           grid = img[h0-150:h0+149, w0-150:w0+149, :]
           
           predicted_boxes, predicted_scores, predicted_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                        feed_dict={self.image: np.expand_dims(grid, axis=0)})
           predicted_boxes = predicted_boxes.squeeze()
           predicted_scores = predicted_scores.squeeze()
           predicted_classes = predicted_classes.squeeze()

           traffic_light = None
           h, w = grid.shape[:2]

           for i in range(predicted_boxes.shape[0]):
                box = predicted_boxes[i]
                score = predicted_scores[i]
                """
                For each boxes detected:
                We check:
                1. If the Trafficlight class has highest score ang geq than a thershold
                3. It is a vertical rectangular shape with xy_ratio smaller than roughly 1/2
                4. Area must be large enough
                """
                if score < self.likelihood: continue
                if predicted_classes[i] != self.coco_traffic_light_type: continue
                x0, y0 = box[1] * w, box[0] * h
                x1, y1 = box[3] * w, box[2] * h
                x0 = int(x0)
                x1 = int(x1)
                y0 = int(y0)
                y1 = int(y1)
                x_diff = x1 - x0
                y_diff = y1 - y0
                xy_ratio = x_diff/float(y_diff)

                if xy_ratio > 0.49: 
                    continue            
                area = np.abs((x1-x0) * (y1-y0)) / float(w*h)
                if area <= 0.001: 
                    continue

                traffic_light = grid[y0:y1, x0:x1]

                if self.debug:
                    cv2.rectangle(img, (x0,y0), (x1,y1), (0, 255, 0), 2)
                    cv2.imshow('Debug', img)
                    cv2.waitKey(1)            

           if traffic_light is not None: break


        if traffic_light is not None:

            """
            For each traffic light box detected
            we converted it to HSV and only look at brightness (V channel)
            Since light is assumed to be bright
            We filtered out light area and dark area, find the median height of them
            By consider the position of light and dark area, we infer light color.
            Assumed it is always Red Yellow Green vertically for traffic light setup
            Here I am consider Yellow as Red so that the car can slow down in advance
            when light turns yellow.
            """
            brightness = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)[:,:,-1]
            light_h,light_w = np.where(brightness >= (brightness.max() - 5))
            light_h_mean = light_h.mean()
            dark_h,dark_w = np.where(brightness <= (brightness.max() - 50))
            dark_h_mean = dark_h.mean()
            total_h = traffic_light.shape[0]
            combined_h = (light_h_mean + (total_h - dark_h_mean))/2
            light_ratio = combined_h / total_h

            if light_ratio < 0.55: # A larger value to include most of YELLOW as RED
                return TrafficLight.RED
            elif light_ratio > 0.60:
                return TrafficLight.GREEN
            else:
                return TrafficLight.YELLOW
  
        return TrafficLight.UNKNOWN