import json
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import hello_helpers.hello_misc as hm

class Replay:

    def __init__(self):
        self.waypoints = {}
        self.start_time = 0
        self.replay_started = False

    def load_waypoints_json(self, json_file):
        """
        Load waypoints from a JSON file.
        """
        with open(json_file, 'r') as f:
            self.waypoints = json.load(f)

    def streamline_waypoints(self):
        """
        Streamline the waypoints by removing redundant points, and* including joint stopped positions as waypoints (this is to time-sync the recorded with the replayed)
        """
        for joint in self.waypoints:
            streamlined = []
            prev_point = None
            for point in self.waypoints[joint]:
                if prev_point is None or point[0] != prev_point[0]:
                    streamlined.append(point)
                prev_point = point
            self.waypoints[joint] = streamlined

    def issue_multipoint_command(self):
        """
        Issue a multipoint command to the robot using the loaded waypoints.
        """

        if not self.waypoints:
            rospy.logwarn("No waypoints loaded. Please load waypoints before issuing a command.")
            return
        
        # for joint in self.waypoints:

if __name__ == "__main__":
    replay = Replay()
    