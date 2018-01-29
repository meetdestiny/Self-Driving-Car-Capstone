#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

from std_msgs.msg import Int32


import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.pose = None #: Current vehicle location + orientation
        self.frame_id = None
        self.previous_car_index = 0 #: Where in base waypoints list the car is
        self.traffic_index = -1 #: Where in base waypoints list the traffic light is
        self.traffic_time_received = rospy.get_time() #: When traffic light info was received

        self.slowdown_coefficient = 1.7
        self.stopped_distance = 0.25
        
        self.loop()
    
    def loop(self):
        """
        Publishes car index and subset of waypoints with target velocities
        """
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rate.sleep()

            if self.base_waypoints is None or self.pose is None or self.frame_id is None:
                continue

            # Where in base waypoints list the car is

            # Get subset waypoints ahead
            next_waypoint = self.get_next_waypoint(self.pose, self.base_waypoints)
            
            m = min(len(self.base_waypoints), next_waypoint + LOOKAHEAD_WPS)
            next_waypoints = self.base_waypoints[next_waypoint:m]
            
            lane = Lane()
            lane.header.frame_id = self.frame_id
            lane.waypoints = next_waypoints
            lane.header.stamp = rospy.Time.now()
            self.final_waypoints_pub.publish(lane)
        
    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose 
        self.frame_id = msg.header.frame_id

    def waypoints_cb(self, lane):
        # TODO: Implement
        self.base_waypoints = lane.waypoints


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_index = msg.data
        self.traffic_time_received = rospy.get_time()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += self.dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    
    # Utility functions
    
    def dl(self, a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    
    #Get closest waypoint. It maybe behind us 
    def get_closest_waypoint(self, pose, waypoints):
    
        min_distance = float('inf')
        closest_waypoint = 0
    
        for i, waypoint in enumerate(waypoints):
            distance = self.dl(pose.position, waypoint.pose.pose.position)
            if distance < min_distance:
                closest_waypoint, min_distance = i, distance
        return closest_waypoint
    
    #Get the nearest waypoint which is ahead of us
    def get_next_waypoint(self, pose, waypoints):
        next_waypoint = self.get_closest_waypoint(pose, waypoints)
        
        heading = math.atan2( (waypoints[next_waypoint].pose.pose.position.y - pose.position.y), (waypoints[next_waypoint].pose.pose.position.x - pose.position.x) )
        
        x = pose.orientation.x
        y = pose.orientation.y
        z = pose.orientation.z
        w = pose.orientation.w
        
        euler_angles = tf.transformations.euler_from_quaternion([x,y,z,w])
        theta = euler_angles[-1]
        angle = math.fabs(theta-heading)
        if angle > math.pi / 4.0:
            next_waypoint += 1

        return next_waypoint
        


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
