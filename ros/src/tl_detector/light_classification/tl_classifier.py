from styx_msgs.msg import TrafficLight
from tl_color_det_cv import TLColorDetectorCV
from tl_color_det_dl import TLColorDetectorDL
import os

class TLClassifier(object):

    def __init__(self, method, debug):
        self.debug = debug

        if method == "comp_vision":
            self.model = TLColorDetectorCV(self.debug)
        if method == "deepl_learning":
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            self.model = TLColorDetectorDL(self.debug, curr_dir+"/models/frozen_inference_graph.pb")
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        return self.model.predict(image)