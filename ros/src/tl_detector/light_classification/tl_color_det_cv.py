import numpy as np
import cv2
from styx_msgs.msg import TrafficLight
import rospy

class TLColorDetectorCV(object):
  def __init__(self, debug):
    self.debug = debug

  def predict(self, image):
    """
    image: cv2.Image (BGR)
    Reference: https://github.com/sol-prog/OpenCV-red-circle-detection.
    Technique: Threshold the input image in order to keep only the red pixels, 
    and search for circles in the result.

    """

    """
    # MedianBlur the image to handle noise
    """
    bgr_image = cv2.medianBlur(image, 3)

    """
    # Convert input image to HSV
    """
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)


    """
    # Threshold the HSV image, keep only the red and yellow pixels
    # Bigger range to cater simulator and traffic light from real logs
    """
    car_yellow_red_range = cv2.inRange(hsv_image, np.array([10, 40, 250]), np.array([31, 150, 255]))
    simulator_upper_red_range = cv2.inRange(hsv_image, np.array([160, 40, 200]), np.array([180, 255, 255]))
    simulator_lower_red_range = cv2.inRange(hsv_image, np.array([0, 160, 200]), np.array([9, 255, 255]))
    simulator_yellow_range = cv2.inRange(hsv_image, np.array([29, 200, 242]), np.array([31, 255, 255]))

    """
    # Combine the above ranges and create a combined filter
    """
    red_hue_filter = cv2.addWeighted(car_yellow_red_range, 1.0, simulator_upper_red_range, 1.0, 0.0)
    red_hue_filter = cv2.addWeighted(red_hue_filter, 1.0, simulator_lower_red_range, 1.0, 0.0)
    red_hue_filter = cv2.addWeighted(red_hue_filter, 1.0, simulator_yellow_range, 1.0, 0.0)


    """
    # Slightly blur the result to avoid false positives
    """
    blurred_image = cv2.GaussianBlur(red_hue_filter, (7,7), 0)

    """
    # use HoughCircles to detect circles
    """
    circles = cv2.HoughCircles(blurred_image, 
                               cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=15,
                               minRadius=3, maxRadius=40)

    """
    # Loop over all detected circles and outline them on the original image (commented out)
    # Hough circles detects only the circular line
    # we want to detect filled circle
    # Therefore we need to calculate a filled ratio
    # If it is above the ratio, consider as a proper filled circle
    """

    prediction = TrafficLight.UNKNOWN
    
    if self.debug:
      cv2.namedWindow('Live', cv2.WINDOW_NORMAL)
      cv2.startWindowThread()

    if circles is not None:
      for current_circle in circles[0,:]:
        """
        #For each circle check the filled ratio
        """
        center_x = int(current_circle[0])
        center_y = int(current_circle[1])
        radius = int(current_circle[2])
        new_radius = radius + 1
        grid = cv2.inRange(blurred_image[center_y-new_radius:center_y+new_radius, center_x-new_radius:center_x+new_radius], 230, 255)
        area = new_radius*new_radius*4
        occupied = cv2.countNonZero(grid)
        ratio = occupied/float(area)
        """
        # if ratio is larger than a threshold report as RED
        """
        if ratio > 0.30:
          prediction = TrafficLight.RED
          
          if self.debug:
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.imshow('Live', image)
            cv2.waitKey(1)
    else:
      if self.debug:
        cv2.imshow('Live', image)
        cv2.waitKey(1)
    return prediction