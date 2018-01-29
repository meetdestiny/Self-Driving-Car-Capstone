from styx_msgs.msg import TrafficLight
import tensorflow as tf
from collections import namedtuple
import numpy as np
import rospy
import cv2

import os

from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


class TLClassifier(object):
    
    Size = namedtuple('Size', ['w', 'h'])


    
    def __init__(self):
        #TODO load classifier
        detector_graph_def = tf.GraphDef()
        with open('/home/student/CarND-Capstone/ros/src/tl_detector/light_classification/frozen_inference_graph.pb', 'rb') as f:
            detector_graph_def.ParseFromString(f.read())
        
        self.sess =  tf.Session()
        tf.import_graph_def(detector_graph_def, name='detector')
        self.detection_input = self.sess.graph.get_tensor_by_name('detector/image_tensor:0')
        self.detection_boxes = self.sess.graph.get_tensor_by_name('detector/detection_boxes:0')
        self.detection_classes = self.sess.graph.get_tensor_by_name('detector/detection_classes:0')
        self.detection_scores = self.sess.graph.get_tensor_by_name('detector/detection_scores:0')
        self.i= 0
        
        self.classifier = load_model('/home/student/Downloads/model/light_classifier_model.h5')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img_expanded = np.expand_dims(image, axis=0)

        rospy.logerr("Detecting traffic light")
        boxes, scores, classes = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                 feed_dict={self.detection_input: img_expanded})
        rospy.logerr("Detected traffic light")

        img_size = (image.shape[1], image.shape[0])

        detected_boxes = boxes[0]
        detected_scores = scores[0]
        detected_classes = classes[0]

        
        #Select those scores where threshold is over 0.9 
        dec_boxes = []
        rospy.logerr("Total boxes:{}".format(str(len(detected_scores))) )
        for i, box in enumerate(detected_boxes[:len(detected_scores)]):
            rospy.logerr("Score is:{} for class {}".format(detected_scores[i],detected_classes[i]) )
            if( detected_scores[i] <0.9):
                continue
            # 10 is class label of traffic lights as per https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
            if( detected_classes[i] != 10) :
                continue
            rospy.logerr("Shape of box:" + str(box.shape))              
            
            roi_box = image[int(img_size[1]*box[0]):int(img_size[1]*box[2]), int(img_size[0]*box[1]):int(img_size[0]*box[3])]
            cv2.imwrite('/disk1/capstone/boximages/{}.jpeg'.format(self.i), roi_box)
            
            prediction_state = self.classifier.predict(self.reshape_image(roi_box))
            
            rospy.logerr("Predicted state : {} ", prediction_state)
            self.i= self.i + 1
        return TrafficLight.UNKNOWN
    
    def reshape_image(self,image):
        x = img_to_array(image.resize((64, 64), Image.ANTIALIAS))
        return x[None, :]

    

