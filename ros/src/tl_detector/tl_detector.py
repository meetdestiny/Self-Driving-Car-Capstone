#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
from math import *
import os


STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.image_number = 0
        self._initialized = True

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1


    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_dist = float('inf')
        closest_wp = 0
        for i in range(len(self.waypoints)):
            dist = math.sqrt((pose.position.x - self.waypoints[i].pose.pose.position.x)**2 + 
                             (pose.position.y - self.waypoints[i].pose.pose.position.y)**2)

            if dist < closest_dist:
                closest_dist = dist
                closest_wp = i

        return closest_wp


    def get_next_waypoint(self, pose):
        closest_wp = self.get_closest_waypoint(pose)
        wp_x = self.waypoints[closest_wp].pose.pose.position.x
        wp_y = self.waypoints[closest_wp].pose.pose.position.y
        heading = math.atan2( (wp_y-pose.position.y), (wp_x-pose.position.x) )
        angle = abs(pose.position.z-heading)
        if angle > math.pi / 4.0:
            closest_wp += 1

        return closest_wp


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        
        gt_image_path = os.path.join('/disk1/capstone/images','{0}.jpg'.format( self.image_number))
        cv2.imwrite(gt_image_path,cv_image )
        self.image_number =  self.image_number + 1
        rospy.loginfo('saved gt data %s',gt_image_path)
        #Get classification
        return self.light_classifier.get_classification(cv_image)


    def universal2car_ref(self, x, y, car_x, car_y, car_yaw):
        shift_x = x - car_x
        shift_y = y - car_y
        x_res = (shift_x * math.cos(-car_yaw) - shift_y * math.sin(-car_yaw))
        y_res = (shift_x * math.sin(-car_yaw) + shift_y * math.cos(-car_yaw))
        return x_res, y_res

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self._initialized):
            return -1, TrafficLight.UNKNOWN;
        light = None
        light_wp = -1
        #light_ahead = self.sensor_dist
        tl_delta = 0.0


        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_next_waypoint(self.pose.pose)
            
            car_x = self.waypoints[car_position].pose.pose.position.x
            car_y = self.waypoints[car_position].pose.pose.position.y
            orientation = self.pose.pose.orientation
            
            quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
            _, _, car_yaw = tf.transformations.euler_from_quaternion(quaternion)
               
            #Find the stopping line which is ahead of current car position and within range 
            min_distance = 200000
            min_index = -1

            for i, stop_line_pos in enumerate(stop_line_positions):
                dist = ((stop_line_pos[0]-car_x)**2 + (stop_line_pos[1]-car_y)**2) ** .5
                if dist < min_distance:
                    tl_car_ref_x, _ = self.universal2car_ref(stop_line_pos[0], stop_line_pos[1], car_x, car_y, car_yaw)
                    if tl_car_ref_x >= -1.4:
                        min_distance = dist
                        min_index = i
            rospy.logerr('min index ={} min_distance={}\n'.format(min_index, min_distance))
          
            # If we have found a stopline which is ahead and within range of consideration, 
            #then find the nearest light to see if we need to actually stop.
            
            if min_index >= 0 and min_distance < 80:
                stopline_pos = stop_line_positions[min_index]
                min_distance = 200000
                min_index = -1
                for i,light_pos in enumerate(self.lights):
                   dist = ((light_pos.pose.pose.position.x - stopline_pos[0])**2 + (light_pos.pose.pose.position.y - stopline_pos[1])**2 ) ** 0.5
                   if dist < min_distance:
                       min_distance = dist
                       min_index = i
                
                light = self.lights[min_index]
                rospy.logerr('min index for light={} min_distance={}\n'.format(min_index, min_distance))
            else:
                light = None
                

        #TODO find the closest visible traffic light (if one exists)

        if light:
            rospy.logerr("found at light  {}".format(str(light)))
            state = self.get_light_state(light)
            return light_wp, state
        #self.waypoints = None
        
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
