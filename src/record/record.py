import math
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint

class Record:
    """
    This class is used to record the "significant" waypoints of a robot's joints during its movement. It stores the joint names and their corresponding waypoints, which include position, velocity, and timestamp information. 
    The class also has flags to indicate whether recording has started or stopped.

    joint_name: [waypoint1, waypoint2, ...]
    waypoint: [position, velocity, time_stamp]

    Note-to-self: the waypoints are recorded when the direction of the joint movement changes. However, the specific velocity/acceleration with which it moves is not recorded (because the velocity is 0 at these points)
    To add such functionality, perhaps record the velocity + acceleration curve throughout the motion trajectory, and record velocity reaches a maximum, and create another wayponit there with that specific velocity.
    """
    
    def __init__(self):
        self.start = False
        self.stop = False

        # self.velocity_threshold = 0.05  # Threshold for detecting significant velocity changes
        # note: in position mode, Stretch position commands are tracked by a trapezoidal motion profile.

        self.waypoints = {} 

    def start_recording(self):
        """
        Start recording the waypoints.
        """
        self.start = True
        self.stop = False

    def stop_recording(self):
        """
        Stop recording the waypoints.
        """
        self.stop = True

    def detect_waypoint(self, joint_name, position, velocity, time_stamp):
        if self.start:
            if joint_name in self.waypoints:
                prev_waypoint = self.waypoints[joint_name][-1]
                prev_joint_direction = math.copysign(1, prev_waypoint[0] if len(self.waypoints[joint_name]) < 2 else prev_waypoint[0] - self.waypoints[joint_name][-2][0])  # Get the direction of the previous joint velocity (don't use velocity itself because it jumps between 0.001 and -0.001 at rest)
                if math.copysign(1, velocity) != prev_joint_direction:
                    self.waypoints[joint_name].append([position, velocity, time_stamp])
            else:
                self.waypoints[joint_name] = []
                self.waypoints[joint_name].append([position, velocity, time_stamp])
        elif self.stop and self.start: #stopped recording, but need to record the last waypoint
            self.start = False
            if joint_name in self.waypoints:
                self.waypoints[joint_name].append([position, velocity, time_stamp])
    
    def get_waypoints(self):
        """
        Get the recorded waypoints.
        """
        return self.waypoints
    
    def clear_waypoints(self):
        """
        Clear the recorded waypoints.
        """
        self.waypoints = {}
    
    def print_waypoints(self):
        """
        Print the recorded waypoints in JSON format.
        """
        import json
        print(json.dumps(self.waypoints, indent=4))

    def save_waypoints(self, file_path):
        """
        Save the recorded waypoints to a JSON file.
        """
        import json
        with open(file_path, 'w') as f:
            json.dump(self.waypoints, f, indent=4)

    def callback(self, data):
        """
        Callback function to process incoming JointTrajectoryPoint messages.
        """
        joint_name = data.joint_names[0]  # Assuming single joint for simplicity
        position = data.positions[0]
        velocity = data.velocities[0]
        time_stamp = data.time_from_start.to_sec()
        self.detect_waypoint(joint_name, position, velocity, time_stamp)

    def main(self):
        """
        Main function to run the recording process.
        """
        rospy.init_node('record_waypoints', anonymous=True)
        rospy.loginfo("Recording waypoints...")

        rospy.Subscriber('/stretch/joint_states', JointTrajectoryPoint, self.callback)

if __name__ == "__main__":
    record = Record()
    record.start_recording()
    record.detect_waypoint("joint1", 0.0, 0.1, 0.0)
    record.detect_waypoint("joint1", 0.5, 0.2, 1.0)
    record.detect_waypoint("joint1", 1.0, -0.1, 2.0)  # change in velocity direction
    record.stop_recording()
    record.print_waypoints()