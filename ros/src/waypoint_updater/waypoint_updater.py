#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
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
MAX_DECEL     = 4.0
STOP_BUFFER   = 5.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity',  TwistStamped,self.current_velocity_cb)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.pose = None #: Current vehicle location + orientation
        self.frame_id = None
        self.previous_car_index = 0 #: Where in base waypoints list the car is
        self.traffic_index = -1 #: Where in base waypoints list the traffic light is
        self.traffic_time_received = rospy.get_time() #: When traffic light info was received
        self.current_velocity = 0
        
        self.slowdown_coefficient = 1.7
        self.stopped_distance = 0.25
        self.braking = False
 
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
            
            tl_dist = self.dl(self.pose.position, self.base_waypoints[self.traffic_index].pose.pose.position)
            min_stopping_dist = self.current_velocity**2 / (2.0 * MAX_DECEL) + STOP_BUFFER
            
            lane = Lane()
            lane.header.frame_id = self.frame_id
            lane.header.stamp = rospy.Time.now()
            
            #Brake only when we have enough space 
            if self.traffic_index == -1 or (not self.braking and tl_dist < min_stopping_dist):
                self.braking = False
                lane.waypoints = next_waypoints
            else:
                self.braking = True
                lane.waypoints = self.get_next_waypoint_velocity(next_waypoints, next_waypoint, self.traffic_index)
            
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
        rospy.logerr("Setting traffic light to {}".format(msg.data))
        self.traffic_index = msg.data
        self.traffic_time_received = rospy.get_time()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass
    
    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

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
        
    def get_next_waypoint_velocity(self, waypoints, current_waypoint, traffic_index):
        final_waypoints = []
        for i in range(, traffic_index):
            index = i % len(waypoints)
            wp = Waypoint()
            wp.pose.pose.position.x  = waypoints[index].pose.pose.position.x
            wp.pose.pose.position.y  = waypoints[index].pose.pose.position.y
            wp.pose.pose.position.z  = waypoints[index].pose.pose.position.z
            wp.pose.pose.orientation = waypoints[index].pose.pose.orientation

            if self.braking:
                # Slowly creep up to light if we have stopped short
                dist = self.distance(self.base_waypoints, index, traffic_index)
                if dist > STOP_BUFFER and self.current_velocity < 1.0:
                    wp.twist.twist.linear.x = 2.0
                elif dist < STOP_BUFFER and self.current_velocity < 1.0:
                    wp.twist.twist.linear.x = 0.0
                else:
                    wp.twist.twist.linear.x = min(self.current_velocity, waypoints[index].twist.twist.linear.x)
            else:
                wp.twist.twist.linear.x = waypoints[index].twist.twist.linear.x
            final_waypoints.append(wp)

        if self.braking:
            # Find the traffic_wp index in final_waypoints to pass to decelerate
            tl_wp = len(final_waypoints)

            # If we are braking set all waypoints passed traffic_wp within LOOKAHEAD_WPS to 0.0
            for i in range(traffic_index, current_waypoint + LOOKAHEAD_WPS):
                index = i % len(waypoints)
                wp = Waypoint()
                wp.pose.pose.position.x  = waypoints[index].pose.pose.position.x
                wp.pose.pose.position.y  = waypoints[index].pose.pose.position.y
                wp.pose.pose.position.z  = waypoints[index].pose.pose.position.z
                wp.pose.pose.orientation = waypoints[index].pose.pose.orientation
                wp.twist.twist.linear.x  = 0.0
                final_waypoints.append(wp)
            final_waypoints = self.decelerate(final_waypoints, current_waypoint, traffic_index)

        return final_waypoints
        
    def decelerate(self, waypoints, current_waypoint, traffic_index):
        last = waypoints[traffic_index]
        last.twist.twist.linear.x = 0.0
        for wp in waypoints[:tl_wp][::-1]:
            dist = self.distance(self.base_waypoints, current_waypoint, last)
            dist = max(0.0, dist - STOP_BUFFER)
            vel  = math.sqrt(2 * self.decel * dist)
            if vel < 1.0:
                vel = 0.0
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
