from styx_msgs.msg import TrafficLight
from tl_color_det_cv import TLColorDetectorCV
from tl_color_det_dl import TLColorDetectorDL

class TLClassifier(object):

    def __init__(self, method):
        self.debug = True

        if method == "comp_vision":
            self.model = TLColorDetectorCV(self.debug)
        else:
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            self.model = TLColorDetectorDL(self.debug, curr_dir+"/models/frozen_model.pb")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        result = self.model.predict(image)
        if result is not None:
            return result
        else:
            return TrafficLight.UNKNOWN
